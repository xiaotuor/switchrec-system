#!/usr/bin/env python3
# app.py — Streamlit 前端（收藏夹持久化 + Two-Tower 延迟加载）

from io import BytesIO
from pathlib import Path
import json, random

import pandas as pd
import streamlit as st

from recommender import (
    load_data, build_feature_matrix, compute_similarity,
    get_top_quality, recommend_by_tags, recommend_hybrid,
)

# ────────────────────────── 常量 ──────────────────────────
DATA_PATH   = Path("data/nintendo_games_enriched.csv")
INTER_CSV   = Path("data/interactions.csv")
PLACEHOLDER = "https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/0.png"

FAV_FILE    = Path("favorites.json")         # ⭐ 持久化收藏夹文件

# ────────────────────────── 收藏夹持久化工具 ──────────────────────────
def _load_fav_set() -> set[int]:
    if FAV_FILE.exists():
        with open(FAV_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def _save_fav_set(fav_set: set[int]):
    with open(FAV_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(fav_set)), f, ensure_ascii=False, indent=2)

# ────────────────────────── 缓存初始化 ──────────────────────────
@st.cache_data
def _init_all():
    df   = load_data(DATA_PATH)
    feat = build_feature_matrix(df)
    sim  = compute_similarity(feat)
    uid  = sorted(pd.read_csv(INTER_CSV)["user_id"].unique().tolist())
    return df, sim, uid

df, sim_df, uid_all = _init_all()

# ────────────────────────── SessionState 初始化 ──────────────────────────
if "fav_set" not in st.session_state:
    st.session_state["fav_set"] = _load_fav_set()     # ⭐ 读文件

fav_set: set[int] = st.session_state["fav_set"]       # type hint

# ────────────────────────── 侧栏过滤 ──────────────────────────
st.sidebar.header("🎛️ 过滤 / 设置")
genre_sel  = st.sidebar.multiselect("按流派过滤",
                                    sorted(df["genre"].unique()),
                                    default=sorted(df["genre"].unique()))
min_rating = st.sidebar.slider("最低评分", 0.0, 5.0, 0.0, 0.1)
min_votes  = st.sidebar.slider("最低评分人数", 0,
                               int(df["ratings_count"].max()), 0, 50)
view_mode  = st.sidebar.radio("视图", ("卡片", "表格"))
top_n      = st.sidebar.slider("展示条数", 5, 40, 12)

# ────────────────────────── 数据过滤 ──────────────────────────
df_flt = (
    df[df["genre"].isin(genre_sel)]
    .query("rating >= @min_rating and ratings_count >= @min_votes")
    .reset_index(drop=True)
)

# ────────────────────────── KPI ──────────────────────────
st.title("🎮 Nintendo Switch 游戏推荐系统")
c1, c2, c3 = st.columns(3)
c1.metric("可用游戏数", len(df_flt))
c2.metric("无标签数量", df_flt["tags"].apply(lambda x: len(x) == 0).sum())
c3.metric("无评分数量", df_flt["rating"].isna().sum())
st.divider()

# ────────────────────────── 渲染辅助 ──────────────────────────
def _render_cards(df_show: pd.DataFrame, prefix: str):
    df_show = df_show.reset_index(drop=True)
    for _, row in df_show.iterrows():
        c1, c2 = st.columns([1, 3])

        # —— 图片 —— #
        img = row.get("background_image") or ""
        if not img or pd.isna(img) or str(img).strip() == "":
            img = PLACEHOLDER
        with c1:
            st.image(img, use_container_width=True)

        # —— 文本 + 收藏 —— #
        with c2:
            st.subheader(row["name"])
            st.write(f"⭐ {row['rating']:.2f}　👥 {int(row['ratings_count'])}")
            st.caption(", ".join(row["tags"][:8]))

            key = f"{prefix}_{row['id']}"
            if row["id"] in fav_set:
                st.success("✅ 已收藏")
            elif st.button("加入收藏", key=key):
                fav_set.add(row["id"])
                _save_fav_set(fav_set)            # ⭐ 保存
                st.rerun()

        st.markdown("---")

def _render(df_show: pd.DataFrame, prefix: str):
    if view_mode == "卡片":
        _render_cards(df_show, prefix)
    else:
        st.dataframe(
            df_show[["name", "genre", "rating", "ratings_count", "released"]]
            .reset_index(drop=True),
            use_container_width=True
        )

# ────────────────────────── 页签布局 ──────────────────────────
tab_hot, tab_tag, tab_sim, tab_tower = st.tabs(
    ["🔥 高质量热门", "🏷️ 标签推荐", "🔍 相似游戏", "🧠 深度召回"]
)

with tab_hot:
    st.subheader("🔥 高质量热门")
    _render(get_top_quality(df_flt, top_n), "hot")

with tab_tag:
    st.subheader("🏷️ 标签推荐")
    q = st.text_input("搜索标签")
    pool = sorted({t for tags in df_flt["tags"] for t in tags
                   if q.lower() in t.lower()})
    tag_sel = st.multiselect("选择标签", pool)
    if tag_sel:
        _render(recommend_by_tags(df_flt.copy(), tag_sel, top_n), "tag")
    else:
        st.info("请选择标签")

with tab_sim:
    st.subheader("🔍 相似游戏")
    game_sel = st.selectbox("选择游戏", df_flt["name"].tolist())
    alpha    = st.slider("Hybrid α", 0.0, 1.0, 0.7, 0.05)
    sim_sub  = sim_df.loc[df_flt.index, df_flt.index]
    if st.button("生成推荐", key="btn_sim"):
        _render(recommend_hybrid(df_flt, sim_sub, game_sel, top_n, alpha), "sim")

with tab_tower:
    st.subheader("🧠 Two-Tower 深度召回")
    st.caption("基于训练好的用户/游戏向量的大规模召回 Top-N")

    # —— 延迟导入，避免无 GPU 本地过慢 —— #
    import tower_utils as tw
    uid_all = tw.get_all_user_ids()

    demo_uid = st.session_state.get("demo_uid", random.choice(uid_all))
    uid_sel  = st.selectbox("选择 user_id", uid_all,
                            index=uid_all.index(demo_uid))

    if st.button("🔀 随机一个用户"):
        st.session_state["demo_uid"] = random.choice(uid_all)
        st.rerun()

    if st.button("深度召回"):
        try:
            recs = tw.recommend_twotower(df_flt, int(uid_sel), top_n)
            _render(recs, "tower")
        except ValueError as e:
            st.error(str(e))

# ────────────────────────── 收藏夹 ──────────────────────────
st.sidebar.header("⭐ 我的收藏夹")
if fav_set:
    fav_df = (
        df[df["id"].isin(fav_set)]
        .loc[:, ["name"]]
        .sort_values("name")
        .rename(columns={"name": "游戏名"})
        .reset_index(drop=True)
    )
    st.sidebar.write(f"已收藏 {len(fav_df)} 款游戏")
    st.sidebar.dataframe(fav_df, use_container_width=True)

    if st.sidebar.button("清空收藏"):
        fav_set.clear()
        _save_fav_set(fav_set)                     # ⭐ 保存
        st.rerun()

    buf = BytesIO()
    fav_df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.sidebar.download_button("下载 CSV", buf.getvalue(),
                               file_name="my_favorites.csv",
                               mime="text/csv")
else:
    st.sidebar.write("暂无收藏")
