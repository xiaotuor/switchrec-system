import pandas as pd
from pathlib import Path

def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        quotechar='"',
        keep_default_na=False
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rankscore"] = pd.to_numeric(df["rankscore"], errors="coerce")

    rank_split = df["rank"].str.split("/", expand=True)
    df["current_rank"] = pd.to_numeric(rank_split[0], errors="coerce")
    df["total_games"] = pd.to_numeric(rank_split[1], errors="coerce")
    df.drop(columns=["rank"], inplace=True)

    df[["min_players", "max_players"]] = (
        df["players"]
          .str.replace("–", "-")            
          .str.split("-", expand=True)
          .apply(pd.to_numeric, errors="coerce")
    )

    date_cols = ["europe_release", "us_release", "japan_release"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["tags"] = (
        df["tags"]
          .str.replace(r"^Tags\s*-\s*", "", regex=True)
          .str.split(",\s*")
          .apply(lambda lst: [t.strip() for t in lst if t.strip()])
    )

    df = df.drop(columns=[c for c in df.columns if df[c].isna().all()])

    return df

df = load_raw("nintendo_games.csv")

keep_columns = [
    "genre", "id", "name", "sinopsis", "tags"
]

df_clean = df[keep_columns]

df_clean.to_csv("nintendo_games_clean.csv", index=False)

print("处理完成，已生成 nintendo_games_clean.csv")