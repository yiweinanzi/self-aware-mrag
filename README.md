# Self-Aware Multimodal RAG 🚀

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**一个基于不确定性估计的自适应多模态检索增强生成系统**

[English](#) | [简体中文](#)

</div>

---

## 📖 简介

Self-Aware-MRAG 是一个创新的多模态检索增强生成（Multimodal RAG）框架，通过**不确定性感知机制**实现智能化的检索决策，显著提升视觉问答任务的性能和效率。

### 🎯 核心特性

- **🧠 不确定性感知**: 基于模型输出的熵值动态判断是否需要检索
- **⚡ 自适应检索**: 仅在模型不确定时触发检索，避免不必要的计算开销
- **📊 多指标评估**: 支持EM、F1、VQA-Score、Recall@5、Faithfulness等7项核心指标
- **🔄 基线对比**: 集成Self-RAG、MR²AG、VisRAG、REVEAL、RagVL、MuRAG等主流方法
- **🎨 灵活配置**: 支持多种检索器（BGE、CLIP）和多模态大模型（Qwen3-VL、LLaVA）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     用户查询 + 图像                        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           多模态大语言模型 (Qwen3-VL)                      │
│              ↓ 生成初始答案                                │
│         计算不确定性 u = Entropy(P(y|q,I))                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
           ┌──────────────┐
           │ u > τ ?      │  ← 阈值判断 (τ=0.35)
           └──┬───────┬───┘
              │       │
         Yes  │       │  No
              ▼       ▼
    ┌─────────────┐  直接返回答案
    │ 检索模块     │
    │  BGE/CLIP   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ 重新生成答案 │
    │ (with docs) │
    └─────────────┘
```

---

## 📦 安装与环境配置

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+推荐)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 24GB+ VRAM (推荐A100)
- **CUDA**: 11.8+
- **存储空间**: 至少100GB

### 1. 克隆仓库

```bash
git clone https://github.com/yiweinanzi/self-aware-mrag.git
cd self-aware-mrag
```

### 2. 创建虚拟环境

```bash
conda create -n multirag python=3.10
conda activate multirag
```

### 3. 安装依赖

```bash
# 安装PyTorch (根据CUDA版本调整)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装FlashRAG及其依赖
cd FlashRAG
pip install -e .
cd ..

# 安装其他依赖
pip install transformers==4.45.0
pip install datasets accelerate sentencepiece
pip install faiss-gpu opencv-python pillow
pip install huggingface_hub wandb
```

### 4. 下载模型和数据

#### 模型下载（需要手动下载）

```bash
# 创建模型目录
mkdir -p models

# 下载以下模型到 models/ 目录:
# 1. Qwen3-VL-8B-Instruct (多模态大模型)
# 2. bge-large-en-v1.5 (文本检索器)
# 3. clip-vit-large-patch14-336 (图像检索器)
```

**模型下载链接**:
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-8B-Instruct)
- [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [CLIP-ViT-Large](https://huggingface.co/openai/clip-vit-large-patch14-336)

#### 数据集准备

```bash
# 下载MRAG-Bench数据集
# 数据集应放置在: FlashRAG/flashrag/data/MRAG-Bench/
```

#### 语料库准备

```bash
# 准备Wikipedia 3M语料库 (需要手动构建或下载)
# 格式: JSONL，包含id、contents、title等字段
# 路径: corpus/corpus_wiki_3m.jsonl
```

### 5. 构建检索索引

```bash
# 激活环境
conda activate multirag

# 构建BGE索引 (耗时约1.5小时)
cd FlashRAG
python tools/rebuild_index_wiki_3m.py \
    --corpus corpus/corpus_wiki_3m.jsonl \
    --output indexes/wiki_3m/bge \
    --faiss-type Flat \
    --batch-size 512
```

---

## 🚀 快速开始

### 1. 运行100样本基线对比实验

```bash
conda activate multirag
cd FlashRAG

# 后台运行实验
nohup python experiments/run_all_baselines_100samples.py > ../run_100samples.log 2>&1 &

# 实时查看日志
tail -f ../run_100samples.log
```

### 2. 阈值敏感性分析

```bash
# 测试不同阈值τ∈{0.25, 0.30, 0.35, 0.40, 0.45}
python experiments/run_threshold_sweep.py
```

### 3. 完整数据集测试

```bash
# 在完整MRAG-Bench数据集上运行 (约2000样本)
python experiments/run_all_baselines_full.py
```

---

## 📊 实验配置

### 默认配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **语料库** | `corpus/corpus_wiki_3m.jsonl` | 纯Wikipedia 3M条目 |
| **索引路径** | `indexes/wiki_3m/bge/` | BGE文本索引 |
| **检索器** | BGE-large-en-v1.5 | 1024维文本嵌入 |
| **MLLM** | Qwen3-VL-8B-Instruct | 多模态大模型 |
| **样本数** | 100 | 快速实验 |
| **不确定性阈值** | τ=0.35 | 检索触发阈值 |
| **批处理大小** | 1 | 批量推理大小 |

### 评估指标（7个核心指标）

1. **EM** (Exact Match): 精确匹配率
2. **F1 Score**: F1分数
3. **VQA-Score**: 视觉问答得分
4. **Recall@5**: Top-5召回率
5. **Faithfulness**: 答案忠实度
6. **Attribution Precision**: 归因精度
7. **Position Bias Score**: 位置偏差得分

---

## 📂 项目结构

```
self-aware-mrag/
├── FlashRAG/                      # 核心框架
│   ├── flashrag/
│   │   ├── pipeline/
│   │   │   └── self_aware_pipeline_qwen3vl.py  # ✅ 核心Pipeline (810行)
│   │   ├── modules/
│   │   │   ├── uncertainty_estimator.py        # ✅ 跨模态不确定性估计 (1115行)
│   │   │   ├── position_aware_fusion.py        # ✅ 位置感知融合 (465行)
│   │   │   ├── attribution.py                  # ✅ 可解释性归因 (687行)
│   │   │   ├── qwen3_vl.py                    # ✅ Qwen3-VL封装 (580行)
│   │   │   ├── bge_reranker.py                # ✅ BGE重排序 (208行)
│   │   │   └── query_reformulation.py          # ✅ 查询增强 (330行)
│   │   ├── retriever/            # 检索器实现
│   │   ├── generator/            # 生成器实现
│   │   ├── dataset/              # 数据集加载
│   │   └── data/
│   │       └── MRAG-Bench/       # MRAG-Bench数据集
│   ├── experiments/
│   │   ├── run_all_baselines_100samples.py    # ✅ 主实验脚本
│   │   ├── baselines/                         # ✅ 7个baseline实现
│   │   └── results_baseline_comparison_100_wiki3m/  # ✅ 实验结果
│   ├── tools/
│   │   ├── build_corpus_wiki_only_3m.py       # 语料库构建
│   │   └── rebuild_index_wiki_3m.py           # 索引构建
│   ├── corpus/                   # 语料库 (2.1GB)
│   │   └── corpus_wiki_3m.jsonl
│   └── indexes/                  # 检索索引 (12GB)
│       └── wiki_3m/bge/
├── models/                       # 模型目录 (36GB, 不上传)
│   ├── Qwen3-VL-8B-Instruct/
│   ├── bge-large-en-v1.5/
│   ├── bge-reranker-v2-m3/
│   └── clip-vit-large-patch14-336/
├── refernce/                     # 参考文档
│   ├── 创新点1-自感知多模态RAG-实施方案.md
│   └── 导师意见版.md
├── _ARCHIVE_2025-11-13/          # 归档目录
│   ├── 01_deprecated_code/       # 废弃代码
│   ├── 02_redundant_experiments/ # 冗余实验
│   ├── 03_historical_docs/       # 历史文档
│   ├── 04_experiment_logs/       # 实验日志
│   └── 05_already_archived/      # 之前归档
├── README.md                     # 项目文档
├── ⚠️项目手册-重要信息-不能删除.md  # 核心配置手册
├── ✅完整修复总结-论文方法-2025-11-01.md  # 技术修复记录
├── 📊优化效果对比-2025-11-03.md   # 最新优化结果
├── 📊导师汇报-自感知多模态RAG-2025-11-03.md  # 最新汇报
├── PROJECT_COMPLETION_REPORT.md  # 项目完成度报告
├── CLEANUP_PLAN.md               # 清理计划
└── requirements.txt              # Python依赖
```

**核心代码**: 6个模块, 3385行
**清理时间**: 2025-11-13
**归档位置**: `_ARCHIVE_2025-11-13/`

---

## 🔬 实验结果

### Self-Aware-MRAG vs Baselines (100样本)

**实验时间**: 2025-10-30  
**数据集**: MRAG-Bench  
**语料库**: Wikipedia 3M  

| 方法 | EM ↑ | F1 ↑ | VQA ↑ | Recall@5 ↑ | Faith. ↑ | Attr. ↑ | Pos.Bias ↓ |
|------|------|------|-------|------------|----------|---------|-----------|
| **Self-Aware-MRAG** | **59.0** | **64.7** | **19.7** | **21.0** | - | - | - |
| Self-RAG | 53.0 | 59.9 | 17.7 | 9.0 | - | - | - |
| RagVL | 53.0 | 59.9 | 17.7 | 9.0 | - | - | - |
| MR²AG | 52.0 | 59.7 | 17.3 | 9.0 | - | - | - |
| VisRAG | 52.0 | 58.9 | 17.3 | 9.0 | - | - | - |
| REVEAL | 51.0 | 58.7 | 17.0 | 9.0 | - | - | - |
| MuRAG | 51.0 | 58.7 | 17.0 | 9.0 | - | - | - |

**关键发现**:
- 🥇 **Self-Aware-MRAG在所有主要指标上均为最佳**
- 📈 **EM提升**: +11.3% (vs 第2名)
- 📈 **F1提升**: +8.0% (vs 第2名)
- 🌟 **Recall@5提升**: +133% (vs 其他方法)，证明检索质量显著优于baseline
- ⚡ **效率相当**: 26.5秒/样本，与baseline相近

*注: Faith.(Faithfulness), Attr.(Attribution Precision), Pos.Bias(Position Bias Score)指标在当前实验配置中未启用*

### 阈值敏感性分析

| 阈值 τ | 检索率 | EM | F1 | VQA-Score |
|--------|--------|-----|-----|-----------|
| 0.25 | XX% | XX.X | XX.X | XX.X |
| 0.30 | XX% | XX.X | XX.X | XX.X |
| **0.35** | **XX%** | **XX.X** | **XX.X** | **XX.X** |
| 0.40 | XX% | XX.X | XX.X | XX.X |
| 0.45 | XX% | XX.X | XX.X | XX.X |

---

## 🛠️ 高级用法

### 自定义实验配置

编辑实验脚本中的配置：

```python
# experiments/run_all_baselines_100samples.py

config = {
    'corpus_path': 'corpus/corpus_wiki_3m.jsonl',
    'index_path': 'indexes/wiki_3m/bge/',
    'retriever_model': '/root/autodl-tmp/models/bge-large-en-v1.5',
    'generator_model': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    'uncertainty_threshold': 0.35,  # 修改阈值
    'sample_size': 100,              # 修改样本数
    'batch_size': 1,
}
```

### 添加新的Baseline方法

参考 `flashrag/pipeline/` 目录下的现有实现：

```python
from flashrag.pipeline import BasicPipeline

class YourCustomPipeline(BasicPipeline):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # 自定义初始化
    
    def run(self, dataset, *args, **kwargs):
        # 自定义推理逻辑
        pass
```

---

## 📈 监控与调试

### 查看实验进度

```bash
# 查看运行日志
tail -f run_100samples.log

# 查看GPU使用情况
nvidia-smi

# 查看进程状态
ps aux | grep python
```

### 常见问题排查

1. **CUDA Out of Memory**
   ```bash
   # 降低batch_size或使用8bit量化
   use_8bit: True
   ```

2. **索引加载失败**
   ```bash
   # 重新构建索引
   python tools/rebuild_index_wiki_3m.py
   ```

3. **数据集路径错误**
   ```bash
   # 检查数据集路径
   ls FlashRAG/flashrag/data/MRAG-Bench/
   ```

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 📚 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{selfaware-mrag-2024,
  title={Self-Aware Multimodal RAG: Uncertainty-Guided Retrieval for Visual Question Answering},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yiweinanzi/self-aware-mrag}}
}
```

---

## 🙏 致谢

- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG): 基础RAG框架
- [Qwen3-VL](https://github.com/QwenLM/Qwen2-VL): 多模态大模型
- [MRAG-Bench](https://github.com/MRAG-Bench): 评测数据集

---

## 📮 联系方式

- **作者**: yiweinanzi
- **邮箱**: 2268867257@qq.com
- **GitHub**: [@yiweinanzi](https://github.com/yiweinanzi)

---

<div align="center">

**⭐ 如果觉得有用，请给个Star！⭐**

Made with ❤️ by yiweinanzi

</div>
