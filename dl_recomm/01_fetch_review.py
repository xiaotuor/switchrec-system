import os, time, json, hashlib
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

RAWG_KEY = "7cf966ab5d9e44b29f7c5377c002c88e"
GAME_CSV = Path("../data/nintendo_games_enriched.csv")
OUT_CSV  = Path("../data/rawg_reviews.csv")

def get_user_id(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:12], 16) % 10_000_000

def fetch_fake_reviews_from_ratings(g: dict):
    game_id = g["id"]
    out = []
    ratings = g.get("ratings", [])
    for r in ratings:
        count = r.get("count", 0)
        for i in range(count):
            out.append({
                "game_id": game_id,
                "username": f"auto_user_{r['title']}_{i}",
                "rating": {
                    "exceptional": 5,
                    "recommended": 4,
                    "meh": 3,
                    "skip": 1
                }.get(r["title"], 3),
                "text": f"[Auto] From RAWG ratings ({r['title']})"
            })
    return out


def main():
    df_games = pd.read_csv(GAME_CSV)
    game_ids = df_games["rawg_id"].dropna().astype(int).unique()

    RAWG_DUMP = Path("../data/rawg_all_games.csv")
    rawg_df = pd.read_csv(RAWG_DUMP)
    rawg_list = [json.loads(x) for x in rawg_df["json"]]

    rawg_dict = {g["id"]: g for g in rawg_list}
    all_rows = []
    for gid in tqdm(game_ids, desc="games"):
        g = rawg_dict.get(gid)
        if g:
            all_rows.extend(fetch_fake_reviews_from_ratings(g))

    df = pd.DataFrame(all_rows)
    df["user_id"] = df["username"].apply(get_user_id)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 写入 {OUT_CSV}, 共 {len(df)} 条交互")


if __name__ == "__main__":
    main()
