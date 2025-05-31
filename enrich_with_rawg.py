#!/usr/bin/env python3
# enrich_with_rawg_fast.py

import time, json
from pathlib import Path

import requests
import requests_cache
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process
from requests.exceptions import HTTPError

# —— 配置 —— 
RAWG_API_KEY = "7cf966ab5d9e44b29f7c5377c002c88e"
CACHE_NAME   = "rawg_cache"             # requests-cache 库缓存名
STATE_FILE   = "rawg_state.json"        # 记录上次抓取的 page
RAWG_DUMP    = "rawg_all_games.csv"   # 全量 RAWG 数据中间文件
INPUT_FILE   = "cleaned_nintendo_games.csv"
OUTPUT_FILE  = "nintendo_games_enriched.csv"

# 初始化缓存（SQLite），过期 1 天
requests_cache.install_cache(CACHE_NAME, expire_after=86400)

def fetch_all_rawg(platform_id=7, page_size=100):
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    state_path = data_dir / STATE_FILE
    dump_path  = data_dir / RAWG_DUMP

    # 如果 dump 已经存在 且 state 文件已经删除 -> 直接加载，不再抓取
    if dump_path.exists() and not state_path.exists():
        df = pd.read_csv(dump_path)
        return [json.loads(x) for x in df["json"].tolist()]

    # 否则走断点续传或初次拉取逻辑
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
        # 写 dump 和 state
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
    """
    1) 先批量拉取 RAWG 全量数据
    2) 从中提取需要的字段（id,name,released,rating,ratings_count,description_raw）
    3) 对齐你的清洗后游戏列表，输出 enriched CSV
    """
    data_dir = Path(__file__).parent / "data"
    # 1) 读本地清洗表
    df_clean = pd.read_csv(data_dir / "nintendo_games_clean.csv", quotechar='"', keep_default_na=False)

    # 2) 拉取 RAWG 数据
    rawg_list = fetch_all_rawg()
    # 3) 提取我们关心的字段并做 DataFrame
    rawg_df = pd.DataFrame([{
        "rawg_id": g["id"],
        "name": g["name"],
        "released": g.get("released"),
        "rating": g.get("rating"),
        "ratings_count": g.get("ratings_count"),
        "background_image": g.get("background_image"),
        "is_nintendo_switch": any(p["platform"]["id"] == 7 for p in g.get("platforms", [])),
        "rawg_tags": [t["name"] for t in (g.get("tags") or [])]  # 这里提取 RAWG tags
    } for g in rawg_list])

    # 4) 首次对齐：尝试 exact match
    merged = pd.merge(df_clean, rawg_df, on="name", how="left")
    # 5) 对于 unmatched，用 rapidfuzz 做模糊匹配
    unmatched = merged[merged["rawg_id"].isna()]
    choices   = rawg_df["name"].tolist()
    for idx, row in tqdm(unmatched.iterrows(), total=len(unmatched), desc="Fuzzy matching"):
        match, score, pos = process.extractOne(row["name"], choices)
        if score > 90:  # 匹配度阈值
            matched_row = rawg_df.iloc[pos]
            for col in ["rawg_id","released","rating","ratings_count"]:
                merged.at[idx, col] = matched_row[col]
    # 6) 合并 tags（原 tags + RAWG tags，去重）
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
    # 6) 保存初步 enriched（不含 description_raw）
    merged.to_csv(data_dir / OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(merged.columns.tolist())
    print(f"初步数据写入：{OUTPUT_FILE}")
    return merged

if __name__ == "__main__":
    df_final = enrich()
    print("✅ 批量拉取并对齐完成")
