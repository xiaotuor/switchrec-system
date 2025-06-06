from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR   = Path("data")
INTER_CSV  = DATA_DIR / "interactions.csv"
EMB_PKL    = DATA_DIR / "item_text_emb.pkl"
CKPT       = Path("dl_recomm/twotower.ckpt")

TXT_DIM = 384
EMB_DIM = 64

def _load_uid_iid_maps():
    df = pd.read_csv(INTER_CSV)
    u2enc = {u: i for i, u in enumerate(df.user_id.unique())}
    i2enc = {g: i for i, g in enumerate(df.game_id.unique())}
    return u2enc, i2enc, df.user_id.unique().tolist()

def recommend_twotower(df_items: pd.DataFrame, user_id: int, topk: int = 10):
    import pickle, torch, torch.nn as nn, pytorch_lightning as pl

    if not hasattr(recommend_twotower, "_asset"):
        uid2enc, iid2enc, uid_list = _load_uid_iid_maps()

        class TwoTower(pl.LightningModule):
            def __init__(self, n_user, n_item, txt_dim=TXT_DIM):
                super().__init__()
                self.u_emb  = nn.Embedding(n_user, EMB_DIM)
                self.i_id_emb = nn.Embedding(n_item, EMB_DIM)
                self.i_txt  = nn.Linear(txt_dim, EMB_DIM)

        model = TwoTower(len(uid2enc), len(iid2enc))
        model = TwoTower.load_from_checkpoint(CKPT, map_location="cpu").eval()

        with open(EMB_PKL, "rb") as f:
            txt_emb = pickle.load(f)  # rawg_id -> np.array(384,)

        item_vecs = np.zeros((len(iid2enc), EMB_DIM), dtype=np.float32)
        with torch.no_grad():
            for rid, enc in iid2enc.items():
                txt = torch.tensor(txt_emb.get(rid, np.zeros(TXT_DIM)),
                                   dtype=torch.float32).unsqueeze(0)
                item_vecs[enc] = (model.i_id_emb(torch.tensor([enc])) + model.i_txt(txt)).numpy()[0]

        recommend_twotower._asset = (model, uid2enc, iid2enc, item_vecs, uid_list)

    model, uid2enc, iid2enc, item_vecs, uid_list = recommend_twotower._asset

    if user_id not in uid2enc:
        raise ValueError(f"user_id {user_id} 不在 interactions.csv！")

    u_vec = model.u_emb(torch.tensor([uid2enc[user_id]])).detach().numpy()[0]
    sims  = item_vecs @ u_vec
    top_idx = sims.argsort()[::-1][:topk]

    rawg_ids = [list(iid2enc.keys())[i] for i in top_idx]
    score_map = dict(zip(rawg_ids, sims[top_idx]))

    out = df_items[df_items["rawg_id"].astype(int).isin(rawg_ids)].copy()
    out["tower_score"] = out["rawg_id"].astype(int).map(score_map)
    return out.sort_values("tower_score", ascending=False)

def get_all_user_ids():
    if hasattr(recommend_twotower, "_asset"):
        return recommend_twotower._asset[-1]
    return _load_uid_iid_maps()[-1]
