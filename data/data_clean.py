import pandas as pd
from pathlib import Path

def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        quotechar='"',
        keep_default_na=False  # 让空字段读成空字符串
    )

    # 统一列名 => 小写+下划线
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 1) price、rankscore 转成 float
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rankscore"] = pd.to_numeric(df["rankscore"], errors="coerce")

    # 2) rank 拆分 current_rank / total_games
    rank_split = df["rank"].str.split("/", expand=True)
    df["current_rank"] = pd.to_numeric(rank_split[0], errors="coerce")
    df["total_games"] = pd.to_numeric(rank_split[1], errors="coerce")
    df.drop(columns=["rank"], inplace=True)

    # 3) players 列变成 min_players / max_players
    df[["min_players", "max_players"]] = (
        df["players"]
          .str.replace("–", "-")            # 兼容特殊破折号
          .str.split("-", expand=True)
          .apply(pd.to_numeric, errors="coerce")
    )

    # 4) 日期列统一转 datetime
    date_cols = ["europe_release", "us_release", "japan_release"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 5) tags 列：去掉前缀、拆分成列表
    df["tags"] = (
        df["tags"]
          .str.replace(r"^Tags\s*-\s*", "", regex=True)
          .str.split(",\s*")
          .apply(lambda lst: [t.strip() for t in lst if t.strip()])
    )

    # 6) 去掉多余空列
    df = df.drop(columns=[c for c in df.columns if df[c].isna().all()])

    return df

df = load_raw("nintendo_games.csv")

# 确认要保留的字段
keep_columns = [
    "genre", "id", "name", "sinopsis", "tags"
]

# 筛选保留字段
df_clean = df[keep_columns]

# 保存处理后的文件
df_clean.to_csv("nintendo_games_clean.csv", index=False)

print("处理完成，已生成 nintendo_games_clean.csv")