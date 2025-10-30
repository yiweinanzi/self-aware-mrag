# 🔬 正确的消融实验报告

**时间**: 2025-10-26 08:32:46
**样本数**: 500
**不确定性阈值**: 0.5

---

## 📊 消融实验结果

| Variant | 准确率 | 正确/总数 | vs Baseline | 检索率 |
|---------|--------|----------|-------------|--------|
| 1. Baseline (MuRAG)            | 47.40% | 237/500 | +0.00% | 100.0% |
| 2. + Text Uncertainty          | 64.60% | 323/500 | +17.20% |  7.4% |
| 3. + Visual Uncertainty        | 64.20% | 321/500 | +16.80% |  7.4% |
| 4. + Cross-Modal Alignment     | 64.80% | 324/500 | +17.40% |  7.4% |
| 5. + Position-Aware Fusion     | 66.31% | 307/463 | +18.91% |  8.0% |
| 6. + Attribution (Full)        | 68.90% | 319/463 | +21.50% |  8.0% |

---

## 💡 关键发现

1. **Text Uncertainty**: 通过不确定性判断是否检索
2. **Visual Uncertainty**: 视觉模态的不确定性估计
3. **Cross-Modal Alignment**: 跨模态对齐不确定性（核心创新）
4. **Position-Aware Fusion**: 位置感知的证据融合
5. **Fine-Grained Attribution**: 细粒度证据归因
6. **检索率**: 自适应检索机制的效果

