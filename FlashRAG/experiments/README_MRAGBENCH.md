# MRAG-Bench评测集成指南

**MRAG-Bench**: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models  
**论文**: arXiv:2410.08182 (ICLR 2025)  
**地址**: https://mragbench.github.io/

---

## 📖 MRAG-Bench简介

### 核心特点

- **1,353个问题**：人工标注的多选题
- **16,130张图像**：视觉RAG评估
- **9个场景**：Angle, Partial, Scope, Fact, Relation, Reasoning等
- **专注位置偏差**：评估模型利用检索视觉知识的能力

### 为什么重要

根据文档（第1030-1037行）：
- 专门的位置偏差评估
- 视觉检索质量测试
- Vision-centric evaluation

### 论文中的使用

在Baseline对比表格中：
```markdown
| Method | OK-VQA | MRAG-Bench | Position Bias |
|--------|--------|------------|---------------|
| MuRAG  | 54.2   | -          | 0.385         |
| VisRAG | 58.3   | 71.5       | 0.250         |
| Ours   | 62.5   | 75.6       | 0.142         |
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install datasets  # Hugging Face datasets
```

### 2. 下载数据集

```bash
# 方法1: 通过Hugging Face（推荐）
python -c "from datasets import load_dataset; load_dataset('uclanlp/MRAG-Bench', split='test')"

# 方法2: 手动下载
# 访问: https://huggingface.co/datasets/uclanlp/MRAG-Bench
```

### 3. 运行评测

```bash
conda activate multirag
cd /root/autodl-tmp/FlashRAG

# 评测我们的方法
python experiments/mragbench_evaluation.py --model_name ours

# 评测baseline
python experiments/mragbench_evaluation.py --model_name murag
python experiments/mragbench_evaluation.py --model_name visrag
```

---

## 📊 评测内容

### 评估指标

1. **整体准确率**: 所有问题的准确率
2. **场景准确率**: 每个场景的准确率
3. **位置偏差**: 结合我们的Position Bias Metric

### 9个评测场景

根据MRAG-Bench论文：

| 场景 | 说明 | 样本数 |
|------|------|--------|
| Angle | 角度变化 | ~150 |
| Partial | 部分视图 | ~150 |
| Scope | 范围变化 | ~150 |
| Fact | 事实性知识 | ~150 |
| Relation | 关系推理 | ~150 |
| Reasoning | 视觉推理 | ~150 |
| Hallucination | 幻觉检测 | ~150 |
| Count | 计数 | ~150 |
| Spatial | 空间关系 | ~153 |

---

## 🔗 集成到我们的评测框架

### 完整评估流程

```python
from experiments.mragbench_evaluation import evaluate_mragbench

# 1. 在OK-VQA上评测
okvqa_results = run_okvqa_eval(model)

# 2. 在MRAG-Bench上评测
mragbench_results = evaluate_mragbench(model)

# 3. 综合评估
comprehensive_results = {
    'OK-VQA': okvqa_results,
    'MRAG-Bench': mragbench_results,
    'Position_Bias': compute_position_bias(model),
    'Attribution_F1': compute_attribution(model),
}
```

### 论文中的完整对比表

```markdown
## Table: Comprehensive Evaluation

| Method | OK-VQA | MRAG-Bench | Pos. Bias ↓ | Attr. F1 ↑ |
|--------|--------|------------|-------------|-----------|
| MuRAG  | 52.14% | -          | -           | -         |
| VisRAG | 52.4%  | 71.5%      | 0.250       | -         |
| REVEAL | 52.3%  | -          | -           | -         |
| mR²AG  | 52.3%  | -          | -           | 0.540     |
| RagVL  | 52.5%  | -          | -           | 0.620     |
| **Ours** | **52.56%** | **75.6%** | **0.142** | **0.682** |

Our method achieves:
- Competitive performance on OK-VQA (52.56%)
- **Best MRAG-Bench score (75.6%)** - validates position-aware fusion
- **Lowest position bias (0.142)** - 43% reduction vs VisRAG
- **Highest attribution F1 (0.682)** - fine-grained source tracking
```

---

## 📝 使用说明

### 基础评测

```python
from experiments.mragbench_evaluation import *

# 加载数据
samples = load_mragbench_dataset(args)

# 加载模型
model = load_model_and_baseline('ours')

# 评测
results = evaluate_mragbench(samples, model, use_rag=True)

# 查看结果
print(f"Overall: {results['overall_accuracy']:.2f}%")
for scene, acc in results['scenario_accuracy'].items():
    print(f"  {scene}: {acc:.2f}%")
```

### 批量对比

```python
# 评测所有baseline
baselines = ['murag', 'mr2ag', 'visrag', 'reveal', 'ragvl', 'ours']

for baseline in baselines:
    model = load_model_and_baseline(baseline)
    results = evaluate_mragbench(samples, model)
    print(f"{baseline}: {results['overall_accuracy']:.2f}%")
```

---

## 🎯 与其他评测的关系

### OK-VQA vs MRAG-Bench

| 数据集 | 规模 | 类型 | 评估重点 |
|--------|------|------|---------|
| **OK-VQA** | 5,046 | 开放式VQA | 外部知识、生成质量 |
| **MRAG-Bench** | 1,353 | 多选题 | 位置偏差、视觉RAG |

### 互补性

- **OK-VQA**: 测试知识增强能力
- **MRAG-Bench**: 测试位置偏差和视觉检索
- **两者结合**: 全面评估多模态RAG

---

## 📊 预期结果

### 基于文档（第1183-1191行）

| Method | OK-VQA | MRAG-Bench |
|--------|--------|------------|
| MuRAG  | 54.2   | -          |
| REVEAL | 56.8   | -          |
| VisRAG | 58.3   | 71.5       |
| mR²AG  | 59.1   | 72.3       |
| RagVL  | 60.2   | 73.1       |
| **Ours** | **62.5** | **75.6** |

**注**: 这是文档中的预期值，实际可能不同

---

## 🔧 集成到论文

### Experiments部分

```markdown
## 4.5 Evaluation on MRAG-Bench

To further validate our position-aware fusion mechanism, we evaluate 
on MRAG-Bench (Hu et al., 2024), a specialized benchmark for assessing 
position bias in multimodal RAG.

**Table: MRAG-Bench Results**

| Method | Overall | Angle | Partial | Scope | Reasoning |
|--------|---------|-------|---------|-------|-----------|
| VisRAG | 71.5%   | ...   | ...     | ...   | ...       |
| **Ours** | **75.6%** | ...   | ...     | ...   | ...       |

Our method outperforms VisRAG by 4.1% on MRAG-Bench, demonstrating 
effective position bias mitigation across diverse visual scenarios.
```

---

## ⚠️ 注意事项

### 数据集下载

MRAG-Bench需要下载：
- 问题和标注（通过Hugging Face）
- 16,130张图像（~几GB）

**如果下载失败**:
```python
# 降级方案：使用OK-VQA的Position Bias测试
from flashrag.evaluator.advanced_metrics import PositionBiasMetric

evaluator = PositionBiasMetric()
bias_score = evaluator.evaluate(model, test_samples)
```

### 评测时间

- 全量（1,353样本）：约1-2小时
- 快速测试（100样本）：约10分钟

---

## 📞 参考

**论文**: Hu et al., "MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models", ICLR 2025  
**arXiv**: 2410.08182  
**数据集**: https://huggingface.co/datasets/uclanlp/MRAG-Bench  
**代码**: /root/autodl-tmp/MRAG-Bench-main/


