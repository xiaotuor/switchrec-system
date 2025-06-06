import time, json
from pathlib import Path

import requests
import requests_cache
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process
from requests.exceptions import HTTPError

RAWG_API_KEY = "7cf966ab5d9e44b29f7c5377c002c88e"
CACHE_NAME   = "rawg_cache"             
STATE_FILE   = "rawg_state.json"        
RAWG_DUMP    = "rawg_all_games.csv"   
INPUT_FILE   = "cleaned_nintendo_games.csv"
OUTPUT_FILE  = "nintendo_games_enriched.csv"

requests_cache.install_cache(CACHE_NAME, expire_after=86400)

def fetch_all_rawg(platform_id=7, page_size=100):
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    state_path = data_dir / STATE_FILE
    dump_path  = data_dir / RAWG_DUMP

    if dump_path.exists() and not state_path.exists():
        df = pd.read_csv(dump_path)
        return [json.loads(x) for x in df["json"].tolist()]

    if dump_path.exists():
        df = pd.read_csv(dump_path)
        all_games = df["json"].apply(json.loads).tolist()
        page = int(state_path.read_text()) if state_path.exists() else 1
    else:
        all_games = []
        page = 1

    session = requests.Session()
    base_url = "https://api.rawg.io/api/games"
    pbar = tqdm(desc="Fetching RAWG pages", unit="page")
    while True:
        params = {"key": RAWG_API_KEY, "platforms": platform_id,
                  "page": page, "page_size": page_size}
        try:
            resp = session.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
        except HTTPError as e:
            if resp.status_code == 404:
                break
            else:
                raise
        data = resp.json().get("results", [])
        if not data:
            break

        all_games.extend(data)
        pd.DataFrame({"json":[json.dumps(x) for x in all_games]}) \
          .to_csv(dump_path, index=False)
        state_path.write_text(str(page))

        pbar.update(1)
        page += 1
        time.sleep(0.2)

    pbar.close()
    state_path.unlink(missing_ok=True)
    return all_games

def enrich():
    data_dir = Path(__file__).parent / "data"
    df_clean = pd.read_csv(data_dir / "nintendo_games_clean.csv", quotechar='"', keep_default_na=False)

    rawg_list = fetch_all_rawg()
    rawg_df = pd.DataFrame([{
        "rawg_id": g["id"],
        "name": g["name"],
        "released": g.get("released"),
        "rating": g.get("rating"),
        "ratings_count": g.get("ratings_count"),
        "background_image": g.get("background_image"),
        "is_nintendo_switch": any(p["platform"]["id"] == 7 for p in g.get("platforms", [])),
        "rawg_tags": [t["name"] for t in (g.get("tags") or [])]
    } for g in rawg_list])

    merged = pd.merge(df_clean, rawg_df, on="name", how="left")
    unmatched = merged[merged["rawg_id"].isna()]
    choices   = rawg_df["name"].tolist()
    for idx, row in tqdm(unmatched.iterrows(), total=len(unmatched), desc="Fuzzy matching"):
        match, score, pos = process.extractOne(row["name"], choices)
        if score > 90:
            matched_row = rawg_df.iloc[pos]
            for col in ["rawg_id","released","rating","ratings_count"]:
                merged.at[idx, col] = matched_row[col]
    new_tags = []
    for _, row in merged.iterrows():
        old_tags = eval(row["tags"]) if pd.notna(row["tags"]) and row["tags"].startswith("[") else []
        rawg_tags = row["rawg_tags"] if isinstance(row["rawg_tags"], list) else []
        combined_tags = list(set(old_tags + rawg_tags))
        new_tags.append(combined_tags)

    merged["tags"] = new_tags

    keep_columns = [
        'genre', 'id', 'name', 'sinopsis', 'tags', 'rawg_id', 'released', 'rating', 'ratings_count', 'background_image', 'is_nintendo_switch'
    ]
    merged = merged[keep_columns]
    merged = merged[merged["is_nintendo_switch"] == True]
    merged.to_csv(data_dir / OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(merged.columns.tolist())
    print(f"初步数据写入：{OUTPUT_FILE}")
    return merged

if __name__ == "__main__":
    df_final = enrich()
    print("✅ 批量拉取并对齐完成")
