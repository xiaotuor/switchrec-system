
```markdown
# 🎮 SwitchRec System

> Nintendo Switch 游戏推荐系统  
> 内容召回 + 深度 Two-Tower 召回 + Streamlit Web 界面

---

## ✨ 功能 Features

| 模块 | 描述 |
|------|------|
| 🔥 高质量热门 | 综合评分 × 热度排序 |
| 🏷️ 标签推荐 | 多标签交集匹配 |
| 🔍 相似游戏 | 内容相似度混合推荐 |
| 🧠 Two-Tower 深度召回 | 用户塔 + 游戏塔 (ID+文本) |
| ⭐ 收藏夹 | 收藏 + 导出 CSV |

---

## 🗂️ 项目结构 Project Structure

SwitchRecSystem/
├─ app.py                      # Streamlit 前端
├─ recommender.py              # 基础推荐逻辑
├─ tower\_utils.py              # Two-Tower 推理模块
├─ dl\_recomm/                  # 深度召回训练代码
│   ├─ 01\_fetch\_review\.py
│   ├─ 02\_build\_interactions.py
│   ├─ 03\_train\_towertower.py
│   └─ twotower.ckpt
├─ data/                       # 数据文件
│   ├─ nintendo\_games\_enriched.csv
│   ├─ item\_text\_emb.pkl
├─ requirements.txt
├─ README.md
└─ .streamlit/config.toml      # Streamlit 配置


---

## ⚙️ 本地运行 Local Run



# 1️⃣ 克隆仓库
git clone https://github.com/xiaotuor/switchrec-system.git
cd switchrec-system

# 2️⃣ (可选) 新建 Conda 虚拟环境
conda create -n switchrec python=3.9
conda activate switchrec

# 3️⃣ 安装依赖
pip install -r requirements.txt

# 4️⃣ 运行 Streamlit App
streamlit run app.py


---

## 🧠 Two-Tower 深度召回训练流程

cd dl_recomm

# 1️⃣ 抓取 RAWG 评论
python 01_fetch_review.py

# 2️⃣ 构建交互矩阵 & 文本向量
python 02_build_interactions.py

# 3️⃣ 训练 Two-Tower
python 03_train_towertower.py


---

## 📜 License

MIT License

```
