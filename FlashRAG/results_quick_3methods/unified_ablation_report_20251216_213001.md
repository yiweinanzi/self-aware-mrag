# 统一消融实验报告

**实验时间**: 20251216_213001
**数据集**: okvqa
**样本数**: 5
**模型**: Qwen3-VL-8B-Instruct
**多模态检索**: BGE(60%) + CLIP(40%)
**不确定性阈值**: 0.43

## 消融变体结果

| 变体 | 描述 | 准确率 | 检索率 | F1 | VQA-Score | Recall@5 | Faithfulness | Attribution |
|------|------|--------|--------|----|----------|----------|-------------|-------------|
| MuRAG | Multimodal Retrieval-Augmented |  40.00% |  100.0% |  40.00% | 3333.33% |  60.00% |  60.00% |  60.00% |
| VisRAG | Visual RAG with BGE reranking |  20.00% |  100.0% |  20.00% | 2000.00% |  60.00% |  40.00% |  40.00% |
| ViDoRAG | Video and Document RAG with mu |   0.00% |  100.0% |   0.00% |   0.00% |  60.00% |  90.00% |  60.00% |

## 🏆 最佳结果

**变体**: MuRAG
**准确率**: 40.00%
**描述**: Multimodal Retrieval-Augmented Generation
✅ **性能良好** 准确率超过40%
