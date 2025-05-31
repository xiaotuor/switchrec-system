import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent / "data"
INPUT = "nintendo_games_enriched.csv"
test_path = data_dir / INPUT
df = pd.read_csv(test_path, encoding="utf-8-sig")

# 1️⃣ 没有标签
num_no_tags = (df["tags"] == "['No tags']").sum()

# 2️⃣ 没有 rating （NaN 或 空）
num_no_rating = df["rating"].isna().sum()

# 总行数
total = len(df)

# 打印
print(f"总游戏数量: {total}")
print(f"没有标签的游戏数量: {num_no_tags} ({num_no_tags/total:.2%})")
print(f"没有 rating 的游戏数量: {num_no_rating} ({num_no_rating/total:.2%})")
