# FlashRAG自感知多模态RAG实验系统

## 🎯 项目概览

本项目实现了一个完整的自感知多模态RAG系统，支持跨模态不确定性估计、位置感知融合和综合评估指标。系统已通过优化实现54%准确率，并支持全样本消融实验。

## 📁 核心文件结构

### 🔧 主要实验脚本（当前使用）
- **`run_unified_ablation.py`** - 🌟 **统一消融实验脚本**（推荐）
  - 支持6个消融变体完整测试
  - 集成综合评估指标系统
  - 支持多GPU并行和SLURM集群
  - 基于54%准确率成功配置

- **`run_full_dataset_ablation_sbatch.sh`** - 🚀 **全样本SLURM提交脚本**
  - 用于完整OK-VQA数据集（5046样本）
  - 自动GPU资源配置
  - 24小时时间限制

### 🧪 测试和验证工具
- **`test_retriever.py`** - 检索器功能测试
- **`test_vqa_improvements.py`** - VQA评估系统测试
- **`evaluation_metrics_checker.py`** - 评估指标验证
- **`enhanced_evaluation.py`** - 增强评估系统

### 📊 实验管理
- **`monitor_experiment.py`** - 实验进度监控工具

## 🚀 快速开始

### 环境激活
```bash
source ~/.bashrc
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments
```

### 小样本验证（推荐先运行）
```bash
python run_unified_ablation.py --max-samples 10 --use-multi-gpu --num-gpus 2
```

### 全样本消融实验
```bash
sbatch run_full_dataset_ablation_sbatch.sh
```

## 📋 消融实验变体

1. **Baseline_MuRAG** - 基础多模态检索方法
2. **Text_Uncertainty_Improved** - 改进版文本不确定性估计
3. **Visual_Uncertainty** - 视觉不确定性估计 + 多模态检索
4. **Cross_Modal_Alignment** - 跨模态对齐不确定性
5. **Position_Aware_Fusion** - 位置感知融合 + 多模态检索
6. **Full_Self_Aware_RAG** - 完整自感知多模态RAG系统

## 📊 综合评估指标

系统实现完整的多维度评估：

### 核心准确率指标
- **Accuracy** - 基于VQA官方标准的准确率
- **VQA_Score** - VQA官方评分
- **F1** - Token-level F1分数

### 检索质量指标
- **Retrieval Rate** - 检索触发率
- **Recall@5** - 前5个文档的检索召回率

### 系统可靠性指标
- **Faithfulness** - 答案与检索文档的一致性
- **Attribution Precision** - 答案归因的精确度
- **Position Bias Score** - 位置偏差得分（越低越好）

## ⚙️ 成功配置参数

基于54%准确率验证的最优配置：
```bash
--dataset okvqa \
--use-improved-estimator \
--text-retrieval-weight 0.6 \
--visual-retrieval-weight 0.4 \
--uncertainty-threshold 0.43 \
--text-weight 0.4 \
--visual-weight 0.3 \
--alignment-weight 0.3 \
--use-multi-gpu --num-gpus 2
```

## 📈 预期性能

- **小样本测试（10-100样本）**: 10-30分钟
- **全样本实验（5046样本）**: 18-24小时（2 GPU）
- **综合评估优化**: 支持，无性能瓶颈
- **准确率目标**: 54%+（已验证）

## 📁 实验结果目录

- `results_unified_ablation/` - 统一消融实验结果
- `results_ablation_100samples_2gpu/` - 100样本测试结果
- `archived/` - 已归档的旧版实验文件

## 🔧 系统特性

### 已实现的关键功能
- ✅ **VQA官方评估标准集成** - 完整的答案标准化和评分
- ✅ **综合评估指标系统** - 8个维度全方位评估
- ✅ **性能优化** - 解决评估瓶颈，支持大数据集
- ✅ **多GPU并行支持** - 自动GPU分布和负载均衡
- ✅ **SLURM集群集成** - 完整的作业调度和监控
- ✅ **实时日志系统** - 详细的实验过程记录

### 技术亮点
- **跨模态不确定性估计** - 文本、视觉、对齐三维度 uncertainty 计算
- **位置感知融合** - 减轻检索位置偏差
- **答案支持度检测** - 智能回退机制
- **文档相关性过滤** - 提高检索质量

## 📊 监控和管理

### 查看实验状态
```bash
# SLURM作业状态
squeue -u zqwang

# 实时日志（假设作业ID为225）
tail -f full_ablation_225.out

# 查看消融实验日志
ls -la results_unified_ablation/
```

### 实验监控工具
```bash
python monitor_experiment.py  # 自动监控脚本
```

## 🗂️ 归档说明

为保持目录整洁，以下文件已归档至 `archived/` 目录：
- 旧版实验脚本
- 过时的测试文件
- 重复的配置文件
- 早期版本的结果文件

## ⚠️ 重要提醒

1. **环境依赖**: 确保激活 `multirag` conda环境
2. **GPU检查**: 运行前确认GPU可用性 `nvidia-smi`
3. **磁盘空间**: 全样本实验需要充足存储空间
4. **时间安排**: 完整实验需要18-24小时
5. **监控建议**: 定期检查实验进度和日志

## 📞 故障排除

### 常见问题及解决方案

1. **CUDA内存不足**
   ```bash
   # 减少批处理大小或使用更少样本
   python run_unified_ablation.py --max-samples 50
   ```

2. **模型加载失败**
   ```bash
   # 检查模型路径
   ls -la /data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct
   ```

3. **评估指标错误**
   ```bash
   # 测试评估系统
   python evaluation_metrics_checker.py
   ```

---

**当前状态**: ✅ 系统已验证，Job 225正在运行全样本消融实验
**最后更新**: 2025-11-25
**维护者**: FlashRAG开发团队