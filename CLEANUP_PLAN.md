# 🧹 项目清理和归档计划

**执行时间**: 2025-11-13  
**目标**: 清理冗余代码和文件，保留核心实现，提升项目可维护性

---

## 📋 清理原则

### 保留标准
1. ✅ **核心创新代码**: 跨模态不确定性估计、位置感知融合
2. ✅ **主实验脚本**: run_all_baselines_100samples.py
3. ✅ **最新文档**: 项目手册、README
4. ✅ **最新实验结果**: 100样本对比结果

### 归档标准
1. 📦 **旧版本代码**: BACKUP、fixed、old等后缀
2. 📦 **重复实验脚本**: threshold_sweep多个版本
3. 📦 **历史文档**: 多个汇报版本、分析报告
4. 📦 **实验日志**: nohup*.out, *.log

### 删除标准
1. ❌ **已归档的代码**: _archived_* 目录
2. ❌ **临时文件**: __pycache__, *.pyc
3. ❌ **未使用的模块**: multimodal_output.py (导师意见删除)

---

## 🗂️ 归档目录结构

```
_ARCHIVE_2025-11-13/
├── 01_deprecated_code/           # 废弃代码
│   ├── pipeline/
│   │   ├── self_aware_mm_pipeline.py
│   │   ├── self_aware_pipeline_fixed.py
│   │   └── self_aware_pipeline_qwen3vl_BACKUP_EM62.py
│   └── modules/
│       ├── uncertainty_estimator_improved.py
│       ├── uncertainty_estimator_seakr_optimized.py
│       ├── multimodal_output.py          # 导师意见删除
│       ├── attribution_llava.py
│       ├── position_fusion_fixed.py
│       └── visual_uncertainty_fixed.py
│
├── 02_redundant_experiments/      # 冗余实验脚本
│   ├── run_threshold_sweep.py (4个版本)
│   ├── run_quick_evaluation.py
│   └── tools/run_threshold_sweep.py
│
├── 03_historical_docs/            # 历史文档
│   ├── 导师汇报完整版.md
│   ├── 📁项目整理总结-2025-10-31.md
│   ├── 🔍性能下降根因分析-2025-11-01.md
│   ├── 🔍性能问题分析-全数据集-2025-11-02.md
│   ├── 📊实验对比-不确定性估计器差异.md
│   ├── 🚨文本不确定性异常分析.md
│   ├── 📋综合优化实施计划-方案C.md
│   └── 🎯Self-RAG-vs-Self-Aware-MRAG深度对比分析.md
│
├── 04_experiment_logs/            # 实验日志
│   ├── nohup_*.out (10个文件)
│   ├── threshold_sweep_results.json
│   └── uncertainty_test_fixed_20251029_104246.json
│
└── 05_already_archived/           # 已归档内容(移动)
    ├── _archived_baselines/
    ├── _archived_code/
    ├── _archived_experiments/
    └── _archived_logs/
```

---

## 🎯 核心保留结构

```
/root/autodl-tmp/
├── FlashRAG/
│   ├── flashrag/
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py                    # FlashRAG基础
│   │   │   ├── self_aware_pipeline_qwen3vl.py # ✅ 核心Pipeline
│   │   │   └── [其他FlashRAG原生pipeline]
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   ├── uncertainty_estimator.py       # ✅ 核心创新1
│   │   │   ├── position_aware_fusion.py       # ✅ 核心创新2
│   │   │   ├── qwen3_vl.py                   # ✅ 模型封装
│   │   │   ├── attribution.py                 # ✅ 可解释性支撑
│   │   │   ├── bge_reranker.py               # ✅ 检索增强
│   │   │   └── query_reformulation.py         # ✅ 查询增强
│   │   └── [其他FlashRAG原生模块]
│   ├── experiments/
│   │   ├── run_all_baselines_100samples.py   # ✅ 主实验
│   │   ├── baselines/                        # ✅ 7个baseline
│   │   └── results_baseline_comparison_100_wiki3m/  # ✅ 最新结果
│   └── tools/
│       ├── build_corpus_wiki_only_3m.py
│       └── rebuild_index_wiki_3m.py
│
├── models/                        # 模型文件(保留)
├── data/                          # 数据集(保留)
│
├── README.md                      # ✅ 项目文档
├── ⚠️项目手册-重要信息-不能删除.md  # ✅ 核心手册
├── ✅完整修复总结-论文方法-2025-11-01.md  # ✅ 技术总结
├── 📊优化效果对比-2025-11-03.md   # ✅ 最新结果
├── 📊导师汇报-自感知多模态RAG-2025-11-03.md  # ✅ 最新汇报
│
├── refernce/                      # ✅ 参考文档(保留)
│   ├── 创新点1-自感知多模态RAG-实施方案.md
│   └── 导师意见版.md
│
└── _ARCHIVE_2025-11-13/           # 📦 归档目录
```

---

## 📊 清理统计

### 代码清理
| 类别 | 文件数 | 操作 |
|------|--------|------|
| Pipeline冗余版本 | 3个 | 归档 |
| Modules冗余版本 | 6个 | 归档 |
| 实验脚本重复 | 5个 | 归档 |
| 已归档目录 | 4个 | 移动到新归档 |

### 文档清理
| 类别 | 文件数 | 操作 |
|------|--------|------|
| 历史分析报告 | 8个 | 归档 |
| 实验日志 | 12个 | 归档 |
| 保留核心文档 | 5个 | 保留 |

---

## ✅ 执行检查清单

- [ ] 创建归档目录结构
- [ ] 移动废弃代码
- [ ] 移动冗余实验脚本
- [ ] 归档历史文档
- [ ] 归档实验日志
- [ ] 清理__pycache__
- [ ] 更新README
- [ ] 验证核心功能可运行

