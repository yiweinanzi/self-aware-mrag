# 📋 Self-Aware-MRAG 综合优化实施计划（方案C）

**日期**: 2025-11-02  
**目标**: 通过4项关键优化，将EM从47.97%提升至≥52%  

---

## 🎯 优化目标

| 指标 | 当前 | 目标 | 提升 |
|------|------|------|------|
| EM | 47.97% | ≥52% | +4.03% |
| 检索率 | 93.57% | 60-70% | -25% |
| 有效检索率 | 93.57% | 40-50% | 质量提升 |

---

## 📅 实施计划（分4步）

### ✅ Step 1: 增加文档相关性过滤（今天，4小时）

**目标**: 借鉴Self-RAG，过滤无关文档

**实施**:
```python
# 在self_aware_pipeline_qwen3vl.py中
# 检索后，生成前，增加过滤

# 新增方法
def _relevance_judgment(self, question: str, document: str, image=None) -> bool:
    """判断文档是否与问题相关（借鉴Self-RAG）"""
    doc_preview = document[:300] + "..." if len(document) > 300 else document
    
    prompt = f"""Task: Is this document relevant to answering the question?

Question: {question}

Document: {doc_preview}

Answer ONLY 'RELEVANT' or 'IRRELEVANT':"""
    
    try:
        response = self.qwen3_vl.generate(
            text=prompt,
            image=None,  # 纯文本判断
            max_new_tokens=5,
            temperature=0.05  # 低温度，更确定
        )
        return 'RELEVANT' in response.upper() and 'IRRELEVANT' not in response.upper()
    except:
        return True  # 默认相关（保守）

# 修改run_single流程
retrieved_docs = self.retriever.search(...)

# ✅ 新增：过滤无关文档
relevant_docs = []
for doc in retrieved_docs[:5]:
    doc_text = doc.get('contents', '')
    if self._relevance_judgment(question, doc_text, image):
        relevant_docs.append(doc_text)

# 回退机制
if not relevant_docs:
    # 检索了但无相关文档，回退到直接回答
    final_answer = self._generate_without_context(question, image)
    return result

# 使用过滤后的文档
fused_docs = self.position_fusion(relevant_docs[:3], ...)
```

**预期效果**:
- 有效文档数: 5个 → 2-3个
- 噪声减少: 40-60%
- EM提升: +1-2%

---

### ✅ Step 2: 调整Threshold（今天，同步）

**目标**: 降低检索率

**实施**:
```python
# run_all_baselines_100samples.py
'uncertainty_threshold': 0.35 → 0.50
```

**预期效果**:
- 检索率: 93.57% → 60-70%
- EM提升: +1-2%

---

### ✅ Step 3: 优化Visual Uncertainty（明天，半天）

**目标**: 增强区分度

**实施**:
```python
# uncertainty_estimator.py中的estimate_visual_uncertainty

# 方案1: 调整公式敏感性
uncertainty = 1.0 - richness_score * 0.8  # 范围[0.2, 1.0]
# 当前: 0.7 - richness * 0.6，范围[0.1, 0.7]

# 方案2: 增加特征维度
features = {
    'norm': feature_norm,
    'std': feature_std,
    'mean_abs': feature_mean_abs,
    'entropy': -torch.sum(softmax * log(softmax)),  # ✅ 新增
    'variance': torch.var(image_features),           # ✅ 新增
}

richness_score = (
    norm_score * 0.3 +
    std_score * 0.3 +
    mean_score * 0.1 +
    entropy_score * 0.2 +   # ✅ 新增
    variance_score * 0.1    # ✅ 新增
)

# 方案3: 动态归一化（基于batch统计）
# （可选，如果方案1-2效果不够）
```

**预期效果**:
- Visual方差: 0.0001 → 0.001+（提升10倍）
- Visual范围: 0.054 → 0.2+（提升4倍）
- 更准确的检索决策

---

### ✅ Step 4: 增加答案验证（后天，半天）

**目标**: 确保答案质量

**实施**:
```python
# self_aware_pipeline_qwen3vl.py

def _support_judgment(self, question: str, answer: str, context: str) -> bool:
    """判断答案是否被上下文支持（借鉴Self-RAG）"""
    prompt = f"""Task: Is the answer supported by the context?

Question: {question}

Context: {context[:400]}

Answer: {answer}

Answer ONLY 'SUPPORTED' or 'NOT SUPPORTED':"""
    
    try:
        response = self.qwen3_vl.generate(
            text=prompt,
            image=None,
            max_new_tokens=5,
            temperature=0.05
        )
        return 'SUPPORTED' in response.upper()
    except:
        return True

# 在run_single中，生成后验证
final_answer = self.qwen3_vl.generate(...)

# ✅ 新增：验证答案
is_supported = self._support_judgment(question, final_answer, fused_docs)

if not is_supported:
    # 答案不被支持，可以：
    # 选项A: 降级到直接回答
    final_answer = self._generate_without_context(question, image)
    # 选项B: 记录但仍使用（用于分析）
    result['support_status'] = 'Not Supported'
else:
    result['support_status'] = 'Supported'
```

**预期效果**:
- 减少幻觉
- 提高可信度
- EM提升: +0.5-1%

---

## 📊 累计预期效果

| 优化项 | EM提升 | 检索率影响 |
|--------|--------|-----------|
| 文档相关性过滤 | +1-2% | - |
| Threshold=0.50 | +1-2% | -25% |
| Visual优化 | +0.5-1% | 更准确 |
| 答案验证 | +0.5-1% | - |
| **总计** | **+3-6%** | **60-70%** |

**最终预期**:
- EM: 47.97% → **51-54%**（达到或超过Self-RAG的51.66%）
- 检索率: 93.57% → **60-70%**
- 速度优势: 仍快1.5-2倍

---

## ⏰ 时间安排

### 今天（2025-11-02，晚上）
- [x] 制定计划
- [ ] **Step 1**: 实现文档相关性过滤（2小时编码 + 2小时测试）
- [ ] **Step 2**: 调整threshold（10分钟）
- [ ] 启动实验（后台运行，11小时）

### 明天（2025-11-03）
- [ ] 检查实验结果
- [ ] **Step 3**: 优化visual uncertainty（4小时）
- [ ] 重新启动实验（后台运行）

### 后天（2025-11-04）
- [ ] 检查实验结果
- [ ] **Step 4**: 增加答案验证（4小时）
- [ ] 最终实验（全数据集，7个方法）

### 周末
- [ ] 分析最终结果
- [ ] 撰写论文相关部分
- [ ] 准备对比表格和图表

---

## 🔍 测试策略

### 快速验证（每次优化后）
```bash
# 20样本快速测试
max_samples = 20
threshold_sweep = [0.35, 0.40, 0.45, 0.50]
# 预计时间：1-2小时
```

### 中等规模测试
```bash
# 100样本中等测试
max_samples = 100
# 预计时间：5-6小时
```

### 全数据集验证
```bash
# 1353样本全量测试
max_samples = None
# 预计时间：11小时/方法
```

---

## 🎯 成功标准

### 必达目标
- [x] EM ≥ 51.66%（至少持平Self-RAG）
- [x] 检索率 60-70%（合理的自适应范围）
- [x] 速度仍快于Self-RAG 1.5倍以上

### 理想目标
- [x] EM ≥ 53%（超越Self-RAG 1.34%）
- [x] F1 ≥ 61%（超越Self-RAG）
- [x] 检索率 50-60%（更精准）

### 最低要求
- [x] EM ≥ 50%（至少恢复到合理水平）
- [x] 检索率 < 80%（避免过度检索）

---

## 📝 备注

### 关键假设
1. **噪声文档是主要问题**：相关性过滤应显著提升
2. **检索率过高**：threshold=0.50应改善
3. **Visual区分度低**：优化后应更准确
4. **答案验证有效**：减少幻觉

### 风险控制
- 每步都做备份
- 快速测试验证效果
- 无效优化及时回退
- 保留最佳版本

### 代码版本管理
```bash
# 当前版本（threshold=0.35，无过滤）
→ _ARCHIVED/2025-11-02_threshold_0.35/

# Step 1+2版本（threshold=0.50，有过滤）
→ _ARCHIVED/2025-11-03_threshold_0.50_with_filter/

# Step 3版本（+ visual优化）
→ _ARCHIVED/2025-11-03_improved_visual/

# Step 4最终版本（+ 答案验证）
→ _ARCHIVED/2025-11-04_final_version/
```

---

**开始执行时间**: 2025-11-02 19:30  
**预计完成时间**: 2025-11-04 18:00  
**总工作量**: ~16小时（编码+测试+实验）

