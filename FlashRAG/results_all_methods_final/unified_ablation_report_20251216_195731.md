# 统一消融实验报告

**实验时间**: 20251216_195731
**数据集**: okvqa
**样本数**: 10
**模型**: Qwen3-VL-8B-Instruct
**多模态检索**: BGE(60%) + CLIP(40%)
**不确定性阈值**: 0.43

## 消融变体结果

| 变体 | 描述 | 准确率 | 检索率 | F1 | VQA-Score | Recall@5 | Faithfulness | Attribution |
|------|------|--------|--------|----|----------|----------|-------------|-------------|
| Self-Aware-MRAG | Our Self-Aware Multimodal RAG  |  50.00% |   90.0% |  23.33% | 1000.00% |  30.00% |  60.00% |  50.00% |
| MuRAG | Multimodal Retrieval-Augmented |  20.00% |  100.0% |  26.67% | 1666.67% |  60.00% |  50.00% |  30.00% |
| VisRAG | Visual RAG with BGE reranking |  20.00% |  100.0% |  20.00% | 2000.00% |  60.00% |  41.67% |  30.00% |
| ViDoRAG | Video and Document RAG with mu |   0.00% |  100.0% |   6.67% |   0.00% |  60.00% |  80.00% |  40.00% |
| mR²AG | multi-step Reflection and Refi |   0.00% |   10.0% |   0.00% |   0.00% |   0.00% |   0.00% |   0.00% |

## 🏆 最佳结果

**变体**: Self-Aware-MRAG
**准确率**: 50.00%
**描述**: Our Self-Aware Multimodal RAG system
🎉 **达到高性能标准！** 准确率超过50%
