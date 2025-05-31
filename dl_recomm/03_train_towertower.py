#!/usr/bin/env python3
"""
Two-Tower 推荐召回  (user tower  +  item tower[id + text])
"""
import pickle, torch, torch.nn as nn, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- paths ----------------
INTER_CSV = Path("../data/interactions.csv")     # user_id, game_id, rating
EMB_PKL   = Path("../data/item_text_emb.pkl")    # {rawg_id: np(768,)}
MODEL_OUT = Path("twotower.ckpt")

# ---------------- hyper ----------------
EMB_DIM = 64
BATCH   = 2048
EPOCHS  = 5
TXT_DIM = 384

# ---------------- Dataset ----------------
class InterDS(Dataset):
    def __init__(self, uid, iid, item_emb_mat):
        self.u = torch.tensor(uid, dtype=torch.long)
        self.i = torch.tensor(iid, dtype=torch.long)
        self.txt = torch.tensor(item_emb_mat[iid], dtype=torch.float32)

    def __len__(self):  return len(self.u)
    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.txt[idx]

# ---------------- Model -----------------
class TwoTower(pl.LightningModule):
    def __init__(self, n_user, n_item):
        super().__init__()
        self.save_hyperparameters()
        self.u_emb   = nn.Embedding(n_user, EMB_DIM)
        self.i_id_emb= nn.Embedding(n_item, EMB_DIM)
        self.i_txt   = nn.Linear(TXT_DIM, EMB_DIM)

    def forward(self, u, i, t):
        u_vec = self.u_emb(u)
        i_vec = self.i_id_emb(i) + self.i_txt(t)
        return (u_vec * i_vec).sum(-1)

    def training_step(self, batch, _):
        u, pos_i, txt = batch
        # 简单负采样: 打乱
        neg_i = pos_i[torch.randperm(len(pos_i))]
        neg_t = txt[torch.randperm(len(txt))]
        pos = self(u, pos_i, txt)
        neg = self(u, neg_i, neg_t)
        loss = torch.mean(torch.nn.functional.softplus(neg - pos))
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# ---------------- main ------------------
def main():
    # 1) 读取交互
    df = pd.read_csv(INTER_CSV)

    # 2) 连续化 id
    uid_map = {u: i for i, u in enumerate(df.user_id.unique())}
    iid_map = {g: i for i, g in enumerate(df.game_id.unique())}
    df["u_enc"] = df.user_id.map(uid_map)
    df["i_enc"] = df.game_id.map(iid_map)

    # 3) 读取文本向量并排成矩阵
    with open(EMB_PKL, "rb") as f:
        emb_dict = pickle.load(f)
    n_item = len(iid_map)
    item_emb_mat = np.zeros((n_item, TXT_DIM), dtype=np.float32)
    for rawg_id, enc in iid_map.items():
        item_emb_mat[enc] = emb_dict.get(rawg_id, np.zeros(TXT_DIM))

    # 4) DataLoader
    ds = InterDS(df.u_enc.values, df.i_enc.values, item_emb_mat)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        num_workers=0, pin_memory=True)

    # 5) Train
    model = TwoTower(n_user=len(uid_map), n_item=n_item)
    trainer = pl.Trainer(max_epochs=EPOCHS,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         log_every_n_steps=50)
    trainer.fit(model, loader)

    trainer.save_checkpoint(MODEL_OUT)
    print(f"✅ 模型已保存: {MODEL_OUT}")

if __name__ == "__main__":
    main()
