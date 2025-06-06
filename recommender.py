import ast
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, quotechar='"', keep_default_na=False)

    df["tags"] = df["tags"].apply(lambda s: ast.literal_eval(s))
    df["rating"]        = pd.to_numeric(df["rating"], errors="coerce")
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
    df["popularity"]    = np.log1p(df["ratings_count"])
    df["text"] = df["enriched_sinopsis"] if "enriched_sinopsis" in df.columns else df["sinopsis"]
    return df

def _vec_tags(df):
    mlb = MultiLabelBinarizer()
    return pd.DataFrame(mlb.fit_transform(df["tags"]), columns=mlb.classes_, index=df.index)

def _vec_text(df, max_feat=800):
    tfidf = TfidfVectorizer(max_features=max_feat, stop_words="english")
    mat   = tfidf.fit_transform(df["text"])
    return pd.DataFrame(mat.toarray(), columns=tfidf.get_feature_names_out(), index=df.index)

def build_feature_matrix(df):
    tags = _vec_tags(df)
    text = _vec_text(df)
    r = (df["rating"]-df["rating"].min())/(df["rating"].max()-df["rating"].min())
    p = (df["popularity"]-df["popularity"].min())/(df["popularity"].max()-df["popularity"].min())
    meta = pd.DataFrame({"scaled_rating": r, "scaled_pop": p}, index=df.index)
    return pd.concat([tags, text, meta], axis=1)

def compute_similarity(feat):
    sim = cosine_similarity(feat.values)
    return pd.DataFrame(sim, index=feat.index, columns=feat.index)

def get_top_quality(df, n=10):
    q = df["rating"].fillna(0)*np.log1p(df["ratings_count"])
    return df.assign(qscore=q).sort_values("qscore", ascending=False).head(n)

def recommend_by_tags(df, tag_list: List[str], n=10):
    score = df["tags"].apply(lambda t: len(set(t)&set(tag_list))/len(tag_list))
    return (df.assign(match=score)
              .query("match>0")
              .sort_values("match", ascending=False)
              .head(n))

def recommend_hybrid(df, sim_df, game, n=10, alpha=0.7):
    if game not in df["name"].values:
        raise ValueError(f"{game} not found")
    idx = df.index[df["name"]==game][0]
    sim = sim_df[idx]
    qual = (df["rating"].fillna(0)*np.log1p(df["ratings_count"]))
    qual = (qual-qual.min())/(qual.max()-qual.min())
    score = alpha*sim + (1-alpha)*qual
    score[idx] = -1
    top = score.sort_values(ascending=False).head(n).index
    return df.loc[top].assign(hybrid_score=score.loc[top])
