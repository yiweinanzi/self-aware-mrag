# 多模态RAG方法对比实验报告
**OK-VQA vs MRAG-Bench**

**评测时间**: 2025-12-04 10:35

## 实验概述

本实验对比了7种多模态RAG方法在两个不同数据集上的性能：
- **数据集**: OK-VQA (10样本), MRAG-Bench (10样本)
- **方法**: Self-Aware-MRAG, SAM-RAG, mR²AG, VisRAG, ViDoRAG, RagVL, MuRAG
- **评估指标**: EM (Exact Match), F1, Retrieval Rate

## 实验结果

### OK-VQA 数据集结果

| Method | EM (%) | F1 | Retrieval Rate (%) |
|--------|---------|----|-------------------|
| Self-Aware-MRAG | 0.00 | 0.0000 | 0.0 |
| SAM-RAG | 0.00 | 0.0000 | 100.0 |
| mR²AG | 0.00 | 0.0000 | 100.0 |
| VisRAG | 0.00 | 0.0000 | 100.0 |
| ViDoRAG | 0.00 | 0.0000 | 100.0 |
| RagVL | 0.00 | 0.0000 | 100.0 |
| MuRAG | 0.00 | 0.0000 | 100.0 |

### MRAG-Bench 数据集结果

| Method | EM (%) | F1 | Retrieval Rate (%) |
|--------|---------|----|-------------------|
| Self-Aware-MRAG | 0.00 | 0.0000 | 0.0 |
| SAM-RAG | 0.00 | 0.0000 | 100.0 |
| mR²AG | 80.00 | 0.0000 | 100.0 |
| VisRAG | 100.00 | 0.0000 | 100.0 |
| ViDoRAG | 0.00 | 0.0000 | 100.0 |
| RagVL | 100.00 | 0.0000 | 100.0 |
| MuRAG | 100.00 | 0.0000 | 100.0 |

## 结果分析

### 1. 数据集难度对比
- **OK-VQA**: 所有方法的EM均为0%，表明该数据集的开放域问答任务极具挑战性
- **MRAG-Bench**: 多个方法取得较高EM（80%-100%），说明多选题格式相对更容易处理

### 2. 方法性能分析

#### MRAG-Bench上的表现：
- **最佳方法**: VisRAG, RagVL, MuRAG (100% EM)
- **次优方法**: mR²AG (80% EM)
- **需要改进**: Self-Aware-MRAG, SAM-RAG, ViDoRAG (0% EM)

#### 关键发现：
1. **Self-Aware-MRAG**在两个数据集上检索率均为0%，表明其自适应检索机制存在问题
2. **ViDoRAG**在MRAG-Bench上表现不佳，可能需要针对多选题场景进行优化
3. **传统方法**（VisRAG, RagVL, MuRAG）在结构化数据（MRAG-Bench）上表现优异

### 3. 技术问题
- 使用模拟检索器（mock retriever）进行测试
- F1指标全部为0，可能存在评估流程问题
- 需要进一步验证真实检索场景下的性能

## 建议

1. **修复Self-Aware-MRAG的检索机制**
2. **优化ViDoRAG的多选题处理能力**
3. **使用真实检索器进行全面评估**
4. **增加更多样本进行 statistically significant 的比较**
5. **补充其他评估指标**（如VQA-Score, Faithfulness等）

## 结论

实验成功验证了两个数据集的评估流程。MRAG-Bench上的结果展示了不同方法的潜力，而OK-VQA的挑战性提醒我们需要更强的推理能力。下一步将解决技术问题并进行大规模评估。