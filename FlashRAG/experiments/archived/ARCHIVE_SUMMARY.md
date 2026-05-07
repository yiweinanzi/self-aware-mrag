# 实验文件归档总结

## 归档时间
2025-11-25 14:11

## 归档原因
为保持主实验目录整洁，将过时和冗余的文件归档，同时保留所有有价值的代码和结果。

## 归档内容

### 📁 archived/obsolete_scripts/
- `archive_old_experiments.py` - 旧版实验归档脚本
- `run_2gpu_real_qwen3vl.py` - 2GPU真实Qwen3VL实验脚本
- `run_real_model_ablation.py` - 真实模型消融实验脚本
- `run_stable_4gpu_ablation.py` - 稳定4GPU消融实验脚本
- `test_unified_ablation.py` - 统一消融测试脚本
- `test_unified_version.py` - 统一版本测试脚本

### 📁 archived/old_configs/
- `EXPERIMENTS_CLEANUP_SUMMARY.md` - 实验清理总结
- `HIGH_PERFORMANCE_DISCOVERY_REPORT.md` - 高性能发现报告
- `README_EXPERIMENTS.md` - 旧版实验说明
- `README_MRAGBENCH.md` - MRAGBench说明
- `README_UNIFIED_EXPERIMENTS.md` - 统一实验说明

### 📁 archived/old_results/
- `ablation_100samples_2gpu.*` - 100样本2GPU消融结果
- `final_ablation_2gpu.*` - 最终2GPU消融结果
- `full_ablation_216.*` 到 `full_ablation_221.*` - 完整消融实验结果
- `slurm-*.out` - SLURM作业输出
- `test_config.json` - 测试配置
- `test_improved_prompt_*.log` - 改进提示测试日志
- `improved_prompt_experiment.log` - 改进提示实验日志
- `run_bge_fixed.sh` - BGE修复脚本
- `results_all_real_qwen3vl/` - 所有真实Qwen3VL结果
- `results_real_model_ablation/` - 真实模型消融结果
- `results_stable_4gpu/` - 稳定4GPU结果

## 保留的核心文件

### 🔧 主要脚本
- `run_unified_ablation.py` - ✅ **统一消融实验脚本**（当前主要使用）
- `run_full_dataset_ablation_sbatch.sh` - ✅ **全样本SLURM提交脚本**
- `run_all_baselines_100samples.py` - 对比实验脚本

### 🧪 测试工具
- `test_retriever.py` - 检索器测试
- `test_vqa_improvements.py` - VQA改进测试
- `evaluation_metrics_checker.py` - 评估指标检查
- `enhanced_evaluation.py` - 增强评估系统
- `test_evaluation_metrics.py` - 评估指标测试
- `quick_test_run.py` - 快速测试运行
- `test_gpu.sh` - GPU测试
- `test_sam_rag.py` - SAM-RAG测试

### 📊 实验管理
- `monitor_experiment.py` - 实验监控工具

### 📁 结果目录
- `results_unified_ablation/` - ✅ **统一消融实验结果**（当前使用）
- `results_ablation_100samples_2gpu/` - 100样本测试结果

### ⚙️ 配置和设置
- `baselines/` - 基准方法配置
- `configs/` - 配置文件
- `setup_environment.sh` - 环境设置脚本
- `run_final_ablation_2gpu.sh` - 最终消融脚本
- `run_full_dataset_ablation.sh` - 全数据集消融脚本

## 重要说明

1. **当前主要脚本**: `run_unified_ablation.py` 是统一的主要实验脚本
2. **全样本实验**: 使用 `run_full_dataset_ablation_sbatch.sh` 提交SLURM作业
3. **活跃实验**: Job 225 正在运行全样本消融实验
4. **监控系统**: 可使用 `monitor_experiment.py` 监控实验进度
5. **评估系统**: 已集成完整的综合评估指标

## 当前状态

- ✅ README.md 已更新为当前系统文档
- ✅ 过时文件已归档至 `archived/` 目录
- ✅ 保留所有有价值的代码和配置
- ✅ 主目录已清理整洁
- 🔄 Job 225 全样本消融实验正在进行

---
**维护者**: FlashRAG开发团队
**最后更新**: 2025-11-25