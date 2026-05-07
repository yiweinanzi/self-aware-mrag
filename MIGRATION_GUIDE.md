# 🚀 Self-Aware Multimodal RAG 项目迁移指南

**更新时间**: 2025-11-21

本指南将引导您将本项目（代码、环境、语料库、索引和模型）迁移到新服务器。遵循“代码走 GitHub，大文件走 rsync”的原则，避免重复下载。

---

## 1. 准备工作 (在旧服务器上)

在开始迁移之前，请确保所有最新的代码和文档已经提交到 GitHub，并导出当前环境依赖。

```bash
# 1) 确保 multirag 环境已激活并导出依赖
conda activate multirag
pip list --format=freeze > requirements_multirag.txt

# 2) 检查 git 状态，确保没有未提交的更改
git status

# 3) 提交并推送必要文件（避免包含大文件）
git add MIGRATION_GUIDE.md requirements_multirag.txt README.md
# 如需提交其他代码改动：git add FlashRAG/ *.py
git commit -m "docs: 更新迁移指南并锁定 multirag 依赖"
git push origin main
```

---

## 2. 代码迁移 (在新服务器上)

建议直接从 GitHub 获取最新代码。

```bash
# 1) 克隆项目（任选其一）
# SSH（推荐）
git clone git@github.com:yiweinanzi/self-aware-mrag.git
# 或 HTTPS
git clone https://github.com/yiweinanzi/self-aware-mrag.git

# 2) 进入项目目录
cd self-aware-mrag
```

---

## 3. 环境迁移 (在新服务器上)

使用 Conda 创建 multirag 环境，并优先使用导出的 requirements_multirag.txt 复现依赖。

```bash
# 1) 创建并激活环境
conda create -n multirag python=3.10 -y
conda activate multirag

# 2) 安装 PyTorch（根据 CUDA 版本选择合适索引）
# 例如 CUDA 12.1：
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu121

# 3) 安装其余依赖（推荐）
pip install -r requirements_multirag.txt

# 4) 安装本地可编辑包（如需要）
# FlashRAG
cd FlashRAG && pip install -e . && cd ..
# LLaVA（可选）
cd LLaVA-main && pip install -e . && cd ..
```

注意: 如果 requirements_multirag.txt 与本地可编辑安装冲突，可先安装 PyTorch 和可编辑包，再用 pip install -r 安装其余依赖。

---

## 4. 数据与模型迁移 (使用 rsync，从旧服务器拉取到新服务器)

以下命令在“新服务器”上执行，使用 rsync 从“旧服务器”拉取大文件。替换为实际账号与路径。

```bash
# 0) 变量与目录准备（在新服务器上）
OLD_USER=your_user
OLD_IP=your.old.server.ip
OLD_ROOT=/root/autodl-tmp    # 旧服务器项目根目录（本仓库根）
NEW_ROOT=$(pwd)               # 新服务器当前仓库目录 self-aware-mrag

# 确保目标目录存在
mkdir -p $NEW_ROOT/models \
         $NEW_ROOT/FlashRAG/corpus \
         $NEW_ROOT/FlashRAG/indexes/wiki_3m/bge \
         $NEW_ROOT/FlashRAG/flashrag/data/MRAG-Bench

# 1) 迁移模型（避免重复下载，包含如下子目录）
#   models/Qwen3-VL-8B-Instruct
#   models/bge-large-en-v1.5
#   models/bge-reranker-v2-m3
#   models/clip-vit-large-patch14-336
#   models/llava-v1.5-7b  (可选)
rsync -avz --partial --progress \
  ${OLD_USER}@${OLD_IP}:$OLD_ROOT/models/ $NEW_ROOT/models/

# 2) 迁移语料库（约2.1GB）
rsync -avz --partial --progress \
  ${OLD_USER}@${OLD_IP}:$OLD_ROOT/FlashRAG/corpus/ $NEW_ROOT/FlashRAG/corpus/

# 3) 迁移索引（约12GB，若带宽受限可选择在新机重建）
rsync -avz --partial --progress \
  ${OLD_USER}@${OLD_IP}:$OLD_ROOT/FlashRAG/indexes/wiki_3m/bge/ $NEW_ROOT/FlashRAG/indexes/wiki_3m/bge/

# 4) 迁移 MRAG-Bench 数据集（如已下载）
rsync -avz --partial --progress \
  ${OLD_USER}@${OLD_IP}:$OLD_ROOT/FlashRAG/flashrag/data/MRAG-Bench/ $NEW_ROOT/FlashRAG/flashrag/data/MRAG-Bench/

# 5) （可选）迁移 Wikipedia 原始数据（若需要在新机重建语料库）
# rsync -avz --partial --progress \
#   ${OLD_USER}@${OLD_IP}:$OLD_ROOT/data/wikipedia/ $NEW_ROOT/data/wikipedia/
```

提示:
- 索引文件大且可再生，若网速慢建议跳过“索引传输”，在新机执行 FlashRAG/tools/rebuild_index_wiki_3m.py 重新构建。
- rsync 可断点续传；中断后重新执行即可。

---

## 5. 验证 (在新服务器上)

```bash
# 1) 环境与 CUDA
conda activate multirag
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'CUDA_VER:', torch.version.cuda)"

# 2) 关键包
python -c "import transformers, faiss; print('Transformers:', transformers.__version__)"
python -c "import flashrag; print('FlashRAG OK')"

# 3) 目录检查
ls -lh FlashRAG/corpus/
ls -lh FlashRAG/indexes/wiki_3m/bge/ || echo '索引未迁移，需重建'
ls -lh models/
ls -lh FlashRAG/flashrag/data/MRAG-Bench/ || echo 'MRAG-Bench 未就位'

# 4) 小样本运行（可选）
cd FlashRAG
python experiments/run_all_baselines_100samples.py  # 默认读取配置
```

如果以上步骤均正常，项目即已完成迁移并可在新服务器运行。🎉

