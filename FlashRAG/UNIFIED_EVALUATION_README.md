# 统一数据集评测系统

## 概述

本系统实现了对4个多模态RAG数据集的统一评测，支持7个核心评测指标。

## 支持的数据集

1. **OK-VQA** - 原始OK-VQA数据集
   - 格式：开放性问题
   - 图像：COCO
   - 路径：`/data1/userdata/zqwang/ACL_data/OK-VQA`

2. **A-OKVQA** - 带推理链的OK-VQA
   - 格式：多选题 + 推理链
   - 图像：COCO
   - 路径：`/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA`

3. **MultiModalQA** - 多模态问答
   - 格式：开放性问题（文本+图像+表格）
   - 图像：自定义
   - 路径：`/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA`

4. **MRAG-Bench** - 多模态RAG评测基准
   - 格式：多选题（A/B/C/D）
   - 场景：9种不同场景
   - 路径：`/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench`

## 7个核心评测指标

1. **EM (Exact Match)** - 精确匹配
2. **F1 Score** - Token级别F1
3. **Recall@5** - 检索召回率
4. **VQA-Score** - VQA官方评分
5. **Faithfulness** - 忠实度（答案与检索文档一致性）
6. **Attribution Precision** - 归因精确度（Bigram级别）
7. **Position Bias Score** - 位置偏差分数

## 系统组件

### 1. 统一数据集加载器
```python
from flashrag.dataset.unified_dataset_loader import load_unified_dataset

# 加载数据集
dataset = load_unified_dataset('okvqa', split='val', max_samples=100)
```

### 2. 统一评测器
```python
from flashrag.evaluator.unified_evaluator import evaluate_unified

# 评测
metrics = evaluate_unified(dataset_name, predictions, references)
```

### 3. 数据集评测管理器
```python
from flashrag.evaluator.dataset_evaluation_manager import DatasetEvaluationManager

# 创建管理器
manager = DatasetEvaluationManager(config)

# 运行评测
results = manager.run_evaluation(datasets=['okvqa', 'mrag-bench'])
```

## 使用方法

### 快速开始

```bash
# 测试系统（无需torch依赖）
python test_unified_evaluation_simple.py

# 运行所有数据集对比
python experiments/run_all_datasets_comparison.py --max-samples 100

# 运行特定数据集评测
python experiments/run_all_datasets_comparison.py --datasets okvqa a-okvqa
```

### 参数说明

- `--max-samples`: 每个数据集的最大样本数（默认：100）
- `--datasets`: 要评测的数据集列表
- `--output-dir`: 输出目录

## 输出结果

### 1. 结果文件
- `all_datasets_results_*.json`: 完整评测结果
- `all_datasets_summary_*.json`: 指标摘要
- `comprehensive_report_*.md`: 详细报告

### 2. 报告内容
- 总体性能对比表
- 各数据集详细分析
- 方法对比总结
- MRAG-Bench场景准确率

## 支持的方法

1. **Self-Aware-MRAG** (Our Method)
2. **SAM-RAG**
3. **mR²AG**
4. **VisRAG**
5. **ViDoRAG** (已替换REVEAL)
6. **RagVL**
7. **MuRAG**

## 注意事项

1. **数据准备**
   - 确保数据集已下载到正确位置
   - OK-VQA数据需要单独准备

2. **依赖安装**
   ```bash
   pip install torch transformers datasets
   pip install faiss-cpu pillow numpy tqdm
   ```

3. **GPU要求**
   - 推荐使用GPU运行
   - 支持多GPU并行

## 示例输出

```
评测数据集: MRAG-BENCH
样本数: 100

MRAG-BENCH 评测结果:
------------------------------------------------------------
  准确率              : 65.00%
  F1 Score            : 0.6500
  检索率              : 85.00%
  Recall@5           : 0.7200
  Faithfulness       : 0.8000
  Attribution Precision: 0.7500
  Position Bias Score: 0.3200

场景准确率:
  Overall            : 65.00%
  Angle              : 70.00%
  Partial            : 60.00%
  ...
```

## 已完成的工作

✅ 1. ViDoRAG已成功替换REVEAL
✅ 2. 4个数据集的统一加载器
✅ 3. 7个核心指标的统一评测
✅ 4. MRAG-Bench官方评测代码集成
✅ 5. 统一的评测管理器
✅ 6. 综合对比实验脚本

## 下一步计划

1. 集成真实的模型推理
2. 添加更多评测指标
3. 支持更多数据集
4. 优化评测速度