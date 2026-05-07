# 归档说明 - 2025-10-30

## 归档时间
2025-10-30 17:45

## 归档原因
这些是早期的实验代码和结果，已被新版本替代。为了保持项目目录整洁，将过时内容归档。

## 归档内容

### 过时的实验目录
- `fixed/`, `fixed_pipeline_test/` - 早期的bug修复版本
- `full_scale_memory/` - 早期大规模实验尝试
- `ultimate_21M/` - 21M数据集实验（已废弃）
- `wiki_cc3m/`, `with_cc3m/`, `with_images/`, `with_retriever/` - 混合数据集实验
- `ablation_500_5M/`, `all_methods_100_3M/` - 早期消融实验
- `multimodal_output_test/` - 测试代码
- `threshold_sweep/` - 旧版阈值实验（现在用`run_threshold_sweep*.py`）

### 旧实验结果
- `results_baseline_comparison_100/` - 第一版100样本对比（564KB）
- `results_baseline_comparison_100_real/` - 第二版100样本对比（924KB）

**问题**: 这两个实验的baseline实现不正确，都是"简单RAG"，没有体现各方法的核心创新

## 当前使用的版本

### 实验脚本
- `/root/autodl-tmp/FlashRAG/experiments/run_all_baselines_100samples.py` - 统一baseline对比
- `/root/autodl-tmp/FlashRAG/experiments/run_threshold_sweep_50_realtime.py` - 阈值扫描

### Baseline实现
- `/root/autodl-tmp/FlashRAG/experiments/baselines/*.py` - 独立baseline实现
- 集成在 `run_all_baselines_100samples.py` 中

### 最新结果
- `/root/autodl-tmp/FlashRAG/experiments/results_baseline_comparison_100_wiki3m/` - 2025-10-30

## 重要提醒

**不要使用这个归档目录中的代码！**

这些代码可能：
- 使用了错误的路径
- 实现不完整或有bug
- 配置过时
- 结果不准确

请使用项目根目录和`FlashRAG/`下的最新代码。

---

**参考文档**: `/root/autodl-tmp/⚠️项目手册-重要信息-不能删除.md`

