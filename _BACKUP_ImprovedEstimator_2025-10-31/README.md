# 备份说明 - ImprovedUncertaintyEstimator版本

**备份时间**: 2025-10-31 23:20  
**备份原因**: 切换到CrossModalUncertaintyEstimator（论文完整实现）前的备份  

---

## 备份内容

### 1. 配置文件
- `run_all_baselines_100samples.py.backup`
  - 包含 `use_improved_estimator: True` 的配置

### 2. Pipeline代码
- `self_aware_pipeline_qwen3vl.py.backup`
  - 完整的pipeline实现

### 3. 实验日志
- `full_dataset_experiment_ImprovedEst.log`
  - Self-Aware-MRAG (ImprovedUncertaintyEstimator) 的完整实验日志
  - 结果: EM 48.7%, F1 55.6%

---

## 当时使用的估计器

**ImprovedUncertaintyEstimator** (简化版):
- **U_text**: 关键词匹配 + 问题长度启发式
- **U_visual**: CLIP特征统计（范数+方差）
- **U_align**: CLIP余弦相似度

**性能**:
- 100样本: EM 62.0%, F1 67.7%
- 1353样本: EM 48.7%, F1 55.6% ❌ (性能下降13.3%)

---

## 切换原因

**问题**: ImprovedUncertaintyEstimator在大规模数据上误判率高
- 关键词方法在复杂问题上失效
- 导致检索决策错误
- 引入噪声或遗漏关键信息

**解决方案**: 切换到CrossModalUncertaintyEstimator
- 使用真正的Gram矩阵 + eigen_score
- 使用真正的Attention variance
- 使用真正的JS散度
- 符合论文承诺

---

## 恢复方法

如果需要恢复到ImprovedUncertaintyEstimator版本：

```bash
cd /root/autodl-tmp/FlashRAG/experiments
cp /root/autodl-tmp/_BACKUP_ImprovedEstimator_2025-10-31/run_all_baselines_100samples.py.backup \
   run_all_baselines_100samples.py

cd /root/autodl-tmp/FlashRAG/flashrag/pipeline
cp /root/autodl-tmp/_BACKUP_ImprovedEstimator_2025-10-31/self_aware_pipeline_qwen3vl.py.backup \
   self_aware_pipeline_qwen3vl.py
```

---

**备份状态**: ✅ 完成  
**下一步**: 切换到CrossModalUncertaintyEstimator并重新运行实验

