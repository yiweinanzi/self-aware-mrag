# 统一消融实验使用指南

## 🎯 概述

`run_unified_ablation.py` 是整合了所有最佳功能的统一消融实验脚本，基于以下成功代码的经验：

- **run_real_model_ablation.py**: 功能最完整的实现
- **run_stable_4gpu_ablation.py**: 稳定的4GPU支持
- **run_2gpu_real_qwen3vl.py**: 成功的2GPU优化

## 🚀 快速开始

### 基本使用

#### 统一版本 (run_unified_ablation.py)
```bash
# 激活环境
source /data0/home/zqwang/miniconda3/bin/activate multirag

# 运行完整消融实验（所有6个变体）
python run_unified_ablation.py

# 指定样本数
python run_unified_ablation.py --max-samples 100

# 指定输出目录
python run_unified_ablation.py --output-dir ./my_results
```

#### 高性能版本 (run_high_performance_ablation.py) - 推荐
基于59%准确率配置优化
```bash
# 运行高性能消融实验（基于59%准确率配置）
python run_high_performance_ablation.py

# 启用多模态检索 (BGE+CLIP)
python run_high_performance_ablation.py --use-multimodal-retrieval

# 2GPU加速
python run_high_performance_ablation.py --use-multi-gpu --num-gpus 2

# 只运行最佳性能变体
python run_high_performance_ablation.py --variants High_Performance_Self_Aware_RAG
```

### 多GPU使用

```bash
# 使用2个GPU
python run_unified_ablation.py --use-multi-gpu --num-gpus 2

# 使用4个GPU
python run_unified_ablation.py --use-multi-gpu --num-gpus 4
```

### 选择特定变体

```bash
# 只运行baseline和full方法
python run_unified_ablation.py --variants Baseline_MuRAG Full_Self_Aware_RAG

# 运行不确定性相关变体
python run_unified_ablation.py --variants Text_Uncertainty Visual_Uncertainty Cross_Modal_Alignment
```

## 📊 实验变体说明

| 变体名 | 描述 | 主要功能 |
|--------|------|----------|
| `Baseline_MuRAG` | 基础多模态检索方法 | 对照基线 |
| `Text_Uncertainty` | 增加文本不确定性估计 | SeaKR扩展到多模态 |
| `Visual_Uncertainty` | 增加视觉不确定性估计 | 视觉注意力方差 |
| `Cross_Modal_Alignment` | 增加跨模态对齐不确定性 | JS散度对齐 |
| `Position_Aware_Fusion` | 增加位置感知融合 | 解决位置偏差 |
| `Full_Self_Aware_RAG` | 完整自感知多模态RAG系统 | 所有功能组合 |

## ⚙️ 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-samples` | None (全部) | 最大样本数 |
| `--output-dir` | `./results_unified_ablation` | 输出目录 |
| `--torch-dtype` | `float16` | PyTorch数据类型 |
| `--max-new-tokens` | 10 | 最大生成token数 |
| `--temperature` | 0.01 | 生成温度 |
| `--retrieval-topk` | 5 | 检索topk |
| `--use-multi-gpu` | False | 使用多GPU |
| `--num-gpus` | 2 | GPU数量 |
| `--variants` | 全部 | 要运行的变体 |

### 最佳配置（基于成功实验）

```python
# 不确定性权重（平衡配置）
'text_weight': 0.35,        # 文本不确定性
'visual_weight': 0.35,      # 视觉不确定性
'alignment_weight': 0.30,   # 对齐不确定性

# 不确定性阈值（P60-P70百分位）
'uncertainty_threshold': 0.35,

# 强制检索（消融实验需要）
'force_retrieval': True,
```

## 📈 结果分析

### 输出文件

实验结束后会在指定目录生成：

```
results_unified_ablation_20251125_XXXXXX.json
├── experiment_info          # 实验配置信息
├── variants_summary         # 各变体结果汇总
└── detailed_results         # 详细结果（前50个样本）
```

### 结果解读

```json
{
  "variants_summary": [
    {
      "variant_name": "Position_Aware_Fusion",
      "variant_description": "增加位置感知融合",
      "accuracy": 0.11,                    # 准确率
      "retrieval_rate": 0.0,               # 检索率
      "execution_time": 120.5,             # 执行时间
      "config": { ... }                    # 配置信息
    }
  ]
}
```

### 性能基准

基于历史实验的最佳结果：

| 变体 | 准确率范围 | 样本数 | 特点 |
|------|------------|--------|------|
| Baseline_MuRAG | 40-50% | 20样本 | 小样本高准确率 |
| Text_Uncertainty | 50-59% | 100样本 | 🏆 **最佳性能** |
| Position_Aware_Fusion | 50-59% | 100样本 | 高准确率 |
| Full_Self_Aware_RAG | 50-59% | 100样本 | 完整功能 |

**重要发现**：
- **Self-Aware-MRAG在MRAG-Bench上达到59%准确率**（EM=0.5900）
- 关键配置：uncertainty_threshold=0.43, use_improved_estimator=True
- 多模态融合检索器：BGE(60%) + CLIP(40%)

## 🔧 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 减少batch size或使用CPU
   python run_unified_ablation.py --torch-dtype float16
   ```

2. **模型路径错误**
   ```bash
   # 检查模型路径
   ls -la /data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct
   ```

3. **检索器初始化失败**
   ```bash
   # 检查索引文件
   ls -la /data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/
   ```

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.35.0
- CUDA >= 11.0
- FAISS
- FlashRAG依赖

## 📝 实验记录

### 成功的实验配置

1. **100样本实验**（推荐快速验证）
   ```bash
   python run_unified_ablation.py --max-samples 100 --variants Baseline_MuRAG Position_Aware_Fusion Full_Self_Aware_RAG
   ```

2. **完整数据集实验**（生产环境）
   ```bash
   python run_unified_ablation.py --use-multi-gpu --num-gpus 2
   ```

3. **4GPU大规模实验**
   ```bash
   python run_unified_ablation.py --use-multi-gpu --num-gpus 4 --torch-dtype bfloat16
   ```

### 预期结果

- **单GPU**: 100样本约15-20分钟
- **2GPU**: 100样本约8-10分钟
- **4GPU**: 100样本约4-6分钟
- **完整数据集**: 2-6小时（取决于GPU数量）

## 📚 相关文件

- **保留的核心文件**:
  - `run_unified_ablation.py` - 统一实验脚本
  - `run_real_model_ablation.py` - 功能参考
  - `run_stable_4gpu_ablation.py` - 4GPU参考
  - `baselines/` - baseline对比方法

- **归档的过时文件**:
  - 位于 `archived/` 目录
  - 包括早期版本、功能不完整或性能较差的实现

## 🤝 贡献指南

如需修改或扩展：

1. 在统一版本中添加新功能
2. 保持向后兼容性
3. 更新此文档
4. 测试新功能的正确性

## 📞 支持

如有问题，请检查：
1. 环境配置是否正确
2. 模型和索引文件是否存在
3. GPU内存是否充足
4. 查看错误日志详细信息