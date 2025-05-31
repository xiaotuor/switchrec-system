#!/usr/bin/env python3
"""
根据评论 CSV 生成交互矩阵 & 预计算游戏文本 embedding
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle

GAME_CSV   = Path("../data/nintendo_games_enriched.csv")
REVIEW_CSV = Path("../data/rawg_reviews.csv")
INTER_CSV  = Path("../data/interactions.csv")
EMB_PKL    = Path("../data/item_text_emb.pkl")

def main():
    df_game = pd.read_csv(GAME_CSV)
    df_rev  = pd.read_csv(REVIEW_CSV)

    # -------- 1) 交互表 --------
    inter = (
        df_rev[["user_id", "game_id", "rating"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    inter.to_csv(INTER_CSV, index=False)
    print(f"交互记录数：{len(inter)}")

    # -------- 2) 文本向量 --------
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = df_game["sinopsis"].fillna("").tolist()
    vecs  = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # 存一个 dict {game_id: np.ndarray(768,)}
    emb_dict = {gid: vec for gid, vec in zip(df_game["rawg_id"].astype(int), vecs)}
    with open(EMB_PKL, "wb") as f:
        pickle.dump(emb_dict, f)
    print(f"✅ 写入 {INTER_CSV}, {EMB_PKL}")

if __name__ == "__main__":
    main()
