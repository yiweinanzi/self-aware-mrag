# MultiModalQA Baseline对比实验报告

**评测时间**: 2025-12-20 23:10:59
**样本数**: 10

## 数据集统计

### 问题类型分布

- TableQ: 3 样本
- Compose(TextQ,TableQ): 3 样本
- Compose(ImageQ,TableQ): 1 样本
- Compare(Compose(TableQ,ImageQ),TableQ): 1 样本
- ImageListQ: 2 样本

### 模态分布

- table: 3 样本
- table,text: 3 样本
- table,image: 1 样本
- image,table: 1 样本
- image: 2 样本

---

## 核心指标对比（7个指标）

| Method | EM | F1 | Recall@5 | VQA | Faith | Attr | PosBias | 时间(s) |
|--------|----|----|----------|-----|-------|------|---------|--------|
| Self-Aware-MRAG | 0.2000 | 0.2800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 3.45 |
| SAM-RAG | 0.5000 | 0.5000 | 0.8000 | 0.0000 | 0.7333 | 0.6000 | 0.4326 | 3.61 |
| mR2AG | 0.5000 | 0.5000 | 0.8000 | 0.0000 | 0.8000 | 0.8000 | 0.4326 | 8.73 |
| VisRAG | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.8333 | 0.8000 | 0.3871 | 3.58 |
| ViDoRAG | 0.6000 | 0.6000 | 0.8000 | 0.0000 | 0.7000 | 0.7000 | 0.4326 | 3.52 |
| RagVL | 0.3000 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 3.28 |
| MuRAG | 0.2000 | 0.2000 | 0.8000 | 0.0000 | 0.6000 | 0.5000 | 0.4326 | 16.59 |

**注**:
- EM: Exact Match (精确匹配)
- F1: Token-level F1
- Recall@5: 检索召回率
- VQA: VQA-Score
- Faith: Faithfulness (忠实度)
- Attr: Attribution Precision (归因精度)
- PosBias: Position Bias Score (位置偏差，越低越好)

---

## MultiModalQA特性分析

### 方法排名

1. **ViDoRAG**: EM=60.00%, 检索率=100.0%
2. **SAM-RAG**: EM=50.00%, 检索率=100.0%
3. **mR2AG**: EM=50.00%, 检索率=100.0%
4. **VisRAG**: EM=50.00%, 检索率=100.0%
5. **RagVL**: EM=30.00%, 检索率=0.0%
6. **Self-Aware-MRAG**: EM=20.00%, 检索率=0.0%
7. **MuRAG**: EM=20.00%, 检索率=100.0%

### 关键发现

- **最佳方法**: ViDoRAG
- MultiModalQA需要处理多种模态信息（文本、表格、图像）
- 包含复合问题类型（Compose）需要多步推理
- 检索系统需要能够跨模态检索相关信息
