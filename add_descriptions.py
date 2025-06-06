#!/usr/bin/env python3
"""
拉取 RAWG description_raw 并补充到 nintendo_games_enriched.csv
"""

import time
import requests
import requests_cache
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# ========== 配置 ==========
RAWG_API_KEY = "7cf966ab5d9e44b29f7c5377c002c88e"
CACHE_NAME   = "rawg_desc_cache"    # 请求缓存名
INPUT_CSV    = "data/nintendo_games_enriched.csv"
OUTPUT_CSV   = "data/nintendo_games_final.csv"
SLEEP        = 0.25                 # 请求间隔（秒）
# ==========================

# 初始化请求缓存（3 天过期）
requests_cache.install_cache(CACHE_NAME, expire_after=259200)

def fetch_description(rawg_id: int) -> Optional[str]:
    """调用 RAWG /games/{id} 获取 description_raw"""
    url = f"https://api.rawg.io/api/games/{rawg_id}"
    params = {"key": RAWG_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("description_raw")
    except Exception:
        return None

def main():
    fp = Path(INPUT_CSV)
    df = pd.read_csv(fp, quotechar='"', keep_default_na=False)

    # 确保存在 description_raw 列
    if "description_raw" not in df.columns:
        df["description_raw"] = ""

    # 筛选出需要补抓的行
    mask = df["rawg_id"].notna() & (df["description_raw"] == "")
    to_fetch = df[mask]

    if to_fetch.empty:
        print("所有 description_raw 已补充，无需再抓！")
    else:
        for idx, row in tqdm(to_fetch.iterrows(),
                             total=len(to_fetch),
                             desc="Fetching descriptions"):
            rawg_str = str(row["rawg_id"]).strip()
            try:
                rawg_id = int(float(rawg_str))   # 兼容 "304247.0" 这类形式
            except ValueError:
                continue                         # 若解析失败就跳过
            descr   = fetch_description(rawg_id) or ""
            df.at[idx, "description_raw"] = descr
            time.sleep(SLEEP)

    # 拼接 enriched_sinopsis
    df["enriched_sinopsis"] = df["sinopsis"].fillna("") + "\n\n" + df["description_raw"].fillna("")

    # 保存最终结果
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 完成详情抓取并输出：{OUTPUT_CSV}")

if __name__=="__main__":
    main()
