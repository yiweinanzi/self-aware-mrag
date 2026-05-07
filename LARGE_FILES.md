# 大文件说明 / Large Files Notice

本项目包含若干大文件，已通过 `.gitignore` 排除在 Git 仓库之外。本文档记录这些文件的位置、用途及获取方式。

---

## 一、语料库与索引（Corpus & Index）

### `FlashRAG/corpus/corpus_wiki_3m.jsonl` — **2.1 GB**

- **用途**：Wikipedia 300 万段落检索语料库，供 FlashRAG 框架构建 BM25/Dense 检索索引使用。
- **格式**：每行一个 JSON 对象，包含 `id`、`title`、`contents` 字段。
- **获取**：可从 FlashRAG 官方仓库或 HuggingFace 数据集下载：
  ```
  https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
  ```
- **本地路径**：`FlashRAG/corpus/corpus_wiki_3m.jsonl`

---

## 二、实验数据集（Experiment Datasets）

### `FlashRAG/experiments/multimodalqa_corpus.jsonl` — **99 MB**

- **用途**：MultiModalQA 数据集的多模态语料库，包含文本、表格、图像描述，用于多模态检索实验。
- **格式**：JSONL，每行包含文档 ID、内容、模态类型等字段。
- **获取**：由 `open_resource/multimodalqa-master` 中的脚本处理生成，原始数据见下方。

---

## 三、开源参考资源（Open Source References）

以下文件位于 `open_resource/` 目录，均为第三方开源项目的压缩包或原始数据，**不属于本项目原创代码**。

| 文件 | 大小 | 说明 | 来源 |
|------|------|------|------|
| `open_resource/multimodalqa-master.zip` | 89 MB | MultiModalQA 完整项目包 | [GitHub](https://github.com/allenai/multimodalqa) |
| `open_resource/Qwen3-VL-main.zip` | 85 MB | Qwen3-VL 视觉语言模型项目包 | [GitHub](https://github.com/QwenLM/Qwen-VL) |
| `open_resource/VisRAG-master.zip` | 20 MB | VisRAG 视觉检索增强生成项目包 | [GitHub](https://github.com/MrLight/VisRAG) |
| `open_resource/ViDoRAG-main.zip` | 13 MB | ViDoRAG 视频文档 RAG 项目包 | [GitHub](https://github.com/Alibaba-NLP/ViDoRAG) |

### MultiModalQA 原始数据集文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `open_resource/multimodalqa-master/multimodalqa-master/dataset/MMQA_texts.jsonl.gz` | 44 MB | MMQA 文本语料（压缩） |
| `open_resource/multimodalqa-master/multimodalqa-master/dataset/MMQA_train.jsonl.gz` | 12 MB | MMQA 训练集（压缩） |
| `open_resource/multimodalqa-master/multimodalqa-master/dataset/MMQA_tables.jsonl.gz` | 9.9 MB | MMQA 表格数据（压缩） |
| `open_resource/VQA-master/Results/OpenEnded_mscoco_train2014_fake_results.json` | 11 MB | VQA 评测用假结果文件 |

---

## 四、参考论文（Reference Papers）

以下 PDF 位于 `refernce/reference_paper/`，为本项目调研阶段参考的学术论文，**不属于本项目原创内容**。

| 文件 | 大小 | 论文 |
|------|------|------|
| `RagVL.pdf` | 26 MB | RagVL: Diagnosing and Healing Multimodal RAG |
| `VISRAG.pdf` | 16 MB | VisRAG: Vision-based RAG on Documents |
| `MRAG-Bench.pdf` | 16 MB | MRAG-Bench: Vision-Centric Evaluation for RAG |
| `FLARE.pdf` | 15 MB | FLARE: Active Retrieval Augmented Generation |
| `REVEAL.pdf` | 12 MB | REVEAL: Retrieval-Augmented Visual-Language Pre-Training |

---

## 五、实验归档结果（Archived Experiment Results）

| 文件 | 大小 | 说明 |
|------|------|------|
| `FlashRAG/experiments/archived/20251125/results_real_ablation/real_ablation_results_20251124_131943.json` | 29 MB | 消融实验完整结果（2025-11-24） |
| `FlashRAG/experiments/archived/20251125/results_final_ablation/final_ablation_results_20251124_210959.json` | 22 MB | 最终消融实验结果（2025-11-24） |
| `FlashRAG/experiments/archived/20251125/results_final_ablation/final_ablation_results_20251124_131341.json` | 22 MB | 最终消融实验结果（2025-11-24） |
| `FlashRAG/experiments/archived/20251125/results_real_ablation/real_ablation_results_20251124_131915.json` | 5.8 MB | 消融实验结果（2025-11-24） |

---

## 六、日志与运行输出（Logs & Outputs）

以下 `.out` / `.err` 文件为 SLURM 作业日志，体积较大且不含代码逻辑，已整体排除。

| 文件 | 大小 | 说明 |
|------|------|------|
| `full_ablation_248.out` | 18 MB | 完整消融实验 SLURM 输出日志 |
| `okvqa_fresh_477.out` | 6.7 MB | OK-VQA 实验 SLURM 输出日志 |
| `FlashRAG/experiments/full_ablation_225.out` | 6.3 MB | 消融实验 SLURM 输出日志 |

---

## 七、其他大文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `open_resource/VisRAG-master/assets/main_figure.png` | 14 MB | VisRAG 论文主图 |
| `LLaVA-main/images/demo_cli.gif` | 9.6 MB | LLaVA 演示动图 |

---

## 如何在本地还原完整环境

1. **克隆仓库**
   ```bash
   git clone git@github.com:yiweinanzi/self-aware-mrag.git
   cd self-aware-mrag
   ```

2. **下载 Wikipedia 语料库**
   ```bash
   mkdir -p FlashRAG/corpus
   # 从 HuggingFace 下载
   huggingface-cli download RUC-NLPIR/FlashRAG_datasets --include "*.jsonl" --local-dir FlashRAG/corpus
   ```

3. **下载 MultiModalQA 数据集**
   ```bash
   # 参考 open_resource/multimodalqa-master 中的说明
   # 或从官方地址下载：https://github.com/allenai/multimodalqa
   ```

4. **构建检索索引**
   ```bash
   cd FlashRAG/experiments
   bash build_multimodalqa_indexes.sh
   ```
