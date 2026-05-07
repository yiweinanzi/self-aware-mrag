# 实验代码说明

**最后更新**: 2025-10-30 (索引构建期间预先准备)

---

## 📁 实验文件列表

### 1. Baseline对比实验
**文件**: `run_all_baselines_100samples.py`
**样本数**: 100
**方法数**: 7个 (Self-Aware-MRAG + 6个baselines)

**对比方法**:
1. Self-Aware-MRAG (Our Method) - 全部创新点启用
2. Self-RAG - 总是检索 + 反思
3. mR²AG - 多轮检索 + 重排
4. VisRAG - 视觉优先
5. REVEAL - 跨模态融合
6. RagVL - 多模态RAG
7. MuRAG - 多路径融合

**7个核心指标**:
- EM (Exact Match)
- F1 (Token-level F1)
- Recall@5 (检索召回率)
- VQA-Score
- Faithfulness (忠实度)
- Attribution Precision (归因精度)
- Position Bias Score (位置偏差)

**运行命令**:
```bash
cd /root/autodl-tmp/FlashRAG/experiments
conda activate multirag
nohup python -u run_all_baselines_100samples.py > baseline_100_wiki3m.log 2>&1 &
```

**预计时间**: 2-3小时 (包含7个方法的完整评测)

---

### 2. 阈值敏感性实验
**文件**: `run_threshold_sweep.py`
**样本数**: 100
**测试阈值**: τ ∈ {0.25, 0.30, 0.35, 0.40, 0.45}

**目的**: 测试不确定性阈值对Self-Aware-MRAG性能的影响

**运行命令**:
```bash
cd /root/autodl-tmp/FlashRAG/experiments
conda activate multirag
nohup python -u run_threshold_sweep.py > threshold_sweep_wiki3m.log 2>&1 &
```

**预计时间**: 1-1.5小时

---

## ✅ 配置已更新（2025-10-30）

### 数据配置
```yaml
语料库: /root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl
  - 来源: Wikipedia (psgs_w100.tsv)
  - 文档数: 3,000,000
  - 大小: 2.1GB

索引: /root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index
  - 模型: BGE-large-en-v1.5
  - 类型: Flat
  - 状态: 构建中 (26%)
```

### 废弃的旧路径（已清理）
- ❌ `corpus/corpus_3m_real.jsonl`
- ❌ `corpus/corpus_3m.jsonl`
- ❌ `indexes/3m_real/`
- ❌ `indexes/3m/`

---

## 📊 实验结果输出

### Baseline对比实验输出
```
experiments/results_baseline_comparison_100_wiki3m/
├── all_results_YYYYMMDD_HHMMSS.json          # 详细预测结果
├── metrics_comparison_YYYYMMDD_HHMMSS.json   # 指标对比
└── COMPARISON_REPORT_YYYYMMDD_HHMMSS.md      # Markdown报告
```

### 阈值敏感性实验输出
```
experiments/results_threshold_sweep_wiki3m/
├── threshold_sweep_results_YYYYMMDD_HHMMSS.json   # 详细结果
└── THRESHOLD_ANALYSIS_REPORT_YYYYMMDD_HHMMSS.md   # 分析报告
```

---

## ⏰ 执行时间线

**当前阶段**: BGE索引构建中 (26% 完成)

### 索引完成后立即执行
1. **P0优先级** - Baseline对比实验
   ```bash
   # 索引完成后检查
   ls -lh /root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/
   
   # 启动实验
   cd /root/autodl-tmp/FlashRAG/experiments
   nohup python -u run_all_baselines_100samples.py > baseline_100_wiki3m.log 2>&1 &
   
   # 监控日志
   tail -f baseline_100_wiki3m.log
   ```

2. **P1优先级** - 阈值敏感性实验
   ```bash
   # Baseline实验完成后启动
   nohup python -u run_threshold_sweep.py > threshold_sweep_wiki3m.log 2>&1 &
   ```

---

## 🔍 实验监控命令

### 检查索引状态
```bash
tail -f /root/autodl-tmp/build_wiki_index_3m.log
ls -lh /root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/
```

### 检查实验进度
```bash
# 查看运行的Python进程
ps aux | grep python | grep -v grep

# 查看实验日志
tail -f /root/autodl-tmp/FlashRAG/experiments/baseline_100_wiki3m.log
tail -f /root/autodl-tmp/FlashRAG/experiments/threshold_sweep_wiki3m.log
```

### 检查GPU使用
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### 检查磁盘空间
```bash
df -h /
```

---

## 🚨 注意事项

1. **等待索引完成**
   - BGE索引必须完成后才能运行实验
   - 预计还需45-60分钟完成

2. **GPU内存管理**
   - Qwen3-VL-8B需要约16GB显存
   - 建议一次只运行一个实验脚本

3. **磁盘空间**
   - 结果文件预计占用100-200MB
   - 保持至少10GB可用空间

4. **日志保存**
   - 所有日志自动保存
   - 建议使用 `nohup` 后台运行

5. **结果备份**
   - 重要结果及时备份到其他位置
   - 实验完成后归档到 `_archived_experiment_reports/`

---

## 📞 快速参考

```bash
# 激活环境
conda activate multirag

# 进入实验目录
cd /root/autodl-tmp/FlashRAG/experiments

# 查看可用实验脚本
ls -lh run_*.py

# 运行实验（后台）
nohup python -u <script_name>.py > <log_name>.log 2>&1 &

# 监控日志
tail -f <log_name>.log

# 检查后台进程
ps aux | grep python | grep -v grep
```

---

**准备完毕！等待索引构建完成后即可开始实验。**

