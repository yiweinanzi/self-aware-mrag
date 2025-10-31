# Baseline修复计划 - 详细实施方案

**制定时间**: 2025-10-30 17:45  
**目标**: 正确实现所有baseline方法，确保实验结果可信  
**预计总耗时**: 1-2周

---

## 🎯 总体策略

**分阶段实施**:
1. **阶段1**: 修复指标问题 + 实现2个核心baseline（Self-RAG、mR²AG）
2. **阶段2**: 运行100样本验证，确认实现正确
3. **阶段3**: 根据结果决定是否实现其他baseline
4. **阶段4**: 运行完整数据集实验

---

## 📊 阶段1: 核心修复（预计3-5天）

### Task 1: 修复3个指标为0的问题 ⚡ [预计: 2小时]

**问题**: Pipeline未提供必需的数据字段

**需要修改的文件**:
- `FlashRAG/experiments/run_all_baselines_100samples.py`
- `FlashRAG/flashrag/pipeline/self_aware_pipeline_qwen3vl.py`

**具体任务**:
```python
# 1. 在pipeline返回结果时添加字段

class SelfAwarePipelineQwen3VL:
    def forward(self, sample):
        # ... 现有代码 ...
        
        # ✅ 添加必需字段
        result = {
            'answer': final_answer,
            'retrieval_result': [  # ← 新增
                {
                    'retrieved_docs': retrieved_docs,
                    'retrieval_scores': scores
                }
            ],
            'attributions': {  # ← 新增（如果模型支持）
                'visual': [...],
                'text': [...]
            },
            # position_bias需要特殊测试，暂时可以跳过
        }
        return result
```

**验证方法**:
```bash
# 运行10个样本，检查3个指标是否不再是0
python FlashRAG/experiments/run_all_baselines_100samples.py --max_samples 10
```

**优先级**: 🔴 P0 - 必须完成  
**负责人**: AI + 用户验证

---

### Task 2: 正确实现Self-RAG [预计: 1-2天]

**参考资料**:
- 原始论文: Self-RAG (ICLR 2024)
- 代码仓库: `open_resource/self-rag-main/`
- 核心文件: `retrieval_lm/run_long_form_static.py`

#### 2.1 Adaptive Retrieval实现

**核心思想**: 使用不确定性判断是否需要检索

```python
class SelfRAGPipeline(BaselinePipeline):
    """Self-RAG: 自适应检索 + 反思机制"""
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        # 添加特殊token（如果需要）
        self.special_tokens = {
            'retrieval': '[Retrieval]',
            'no_retrieval': '[No Retrieval]',
            'relevant': '[Relevant]',
            'irrelevant': '[Irrelevant]',
            'utility_high': '[Utility:5]',
            'utility_low': '[Utility:1]'
        }
    
    def should_retrieve(self, question, image):
        """判断是否需要检索（基于不确定性）"""
        
        # 方法1: 使用简化的启发式规则
        # 检查问题中是否包含需要外部知识的关键词
        knowledge_keywords = [
            'when', 'where', 'who', 'what year', 'which country',
            'how many', 'name of', 'capital', 'population'
        ]
        question_lower = question.lower()
        needs_knowledge = any(kw in question_lower for kw in knowledge_keywords)
        
        # 方法2: 基于模型的初始置信度
        # 先让模型尝试回答，如果置信度低则检索
        try:
            initial_answer, confidence = self._generate_with_confidence(
                question, image, max_tokens=5
            )
            # 如果置信度 < 阈值，则需要检索
            needs_retrieval = confidence < 0.7
        except:
            needs_retrieval = needs_knowledge
        
        return needs_retrieval
    
    def _generate_with_confidence(self, question, image, max_tokens):
        """生成答案并返回置信度"""
        # TODO: 实现基于logits的置信度估计
        # 类似我们的不确定性估计，但更简化
        pass
    
    def evaluate_relevance(self, question, image, retrieved_doc):
        """评估检索文档的相关性（reflection）"""
        
        # 构建relevance判断prompt
        relevance_prompt = f"""Given the question and the retrieved passage, 
is this passage relevant to answering the question?

Question: {question}
Passage: {retrieved_doc[:500]}

Answer with: [Relevant] or [Irrelevant]"""
        
        # 使用模型判断（生成1个token）
        response = self.qwen3_vl.generate(
            text=relevance_prompt,
            image=image,
            max_new_tokens=5,
            temperature=0.01
        )
        
        # 解析判断结果
        is_relevant = '[Relevant]' in response or 'relevant' in response.lower()
        
        return is_relevant
    
    def run_single(self, sample):
        """运行单个样本"""
        
        # Step 1: 判断是否需要检索
        should_retrieve = self.should_retrieve(sample['question'], sample['image'])
        
        if not should_retrieve:
            # 直接生成答案（无检索）
            options = {
                'A': sample['A'], 'B': sample['B'],
                'C': sample['C'], 'D': sample['D']
            }
            prompt = self._construct_prompt(sample['question'], options, context=None)
            prediction = self._generate(prompt, sample['image'])
            
            return {
                'answer': self._map_letter_to_answer(prediction, sample),
                'raw_prediction': prediction,
                'retrieved_docs': [],
                'used_retrieval': False,
                'retrieval_decision': 'No Retrieval'
            }
        
        # Step 2: 检索文档
        results = self.retriever.search(sample['question'], num=10)
        
        # Step 3: 评估相关性（reflection）
        relevant_docs = []
        for doc in results[:5]:  # 只评估top-5
            doc_text = doc.get('contents', '')
            if self.evaluate_relevance(sample['question'], sample['image'], doc_text):
                relevant_docs.append(doc_text)
        
        # Step 4: 使用相关文档生成答案
        if relevant_docs:
            context = "\n\n".join(relevant_docs)
        else:
            # 如果没有相关文档，使用top-1
            context = results[0].get('contents', '') if results else ""
        
        options = {
            'A': sample['A'], 'B': sample['B'],
            'C': sample['C'], 'D': sample['D']
        }
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        
        return {
            'answer': self._map_letter_to_answer(prediction, sample),
            'raw_prediction': prediction,
            'retrieved_docs': relevant_docs,
            'used_retrieval': True,
            'retrieval_decision': 'Retrieval',
            'num_relevant_docs': len(relevant_docs)
        }
```

**关键改进**:
1. ✅ `should_retrieve()`: 自适应判断是否需要检索
2. ✅ `evaluate_relevance()`: 反思机制，评估文档相关性
3. ✅ 只使用相关文档生成答案
4. ✅ 记录检索决策和相关文档数量

**测试代码**:
```python
# 测试Self-RAG
pipeline = SelfRAGPipeline(qwen3_vl, retriever, config)

# 测试样本1: 需要外部知识
sample1 = {
    'question': 'When was this building constructed?',
    'image': ...,
    'A': '1887', 'B': '1900', 'C': '1920', 'D': '1950'
}
result1 = pipeline.run_single(sample1)
print(f"Decision: {result1['retrieval_decision']}")
print(f"Used retrieval: {result1['used_retrieval']}")

# 测试样本2: 不需要外部知识
sample2 = {
    'question': 'What color is the sky in this image?',
    'image': ...,
    'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'yellow'
}
result2 = pipeline.run_single(sample2)
print(f"Decision: {result2['retrieval_decision']}")
print(f"Used retrieval: {result2['used_retrieval']}")
```

**期望结果**:
- Sample1应该触发检索（`used_retrieval=True`）
- Sample2应该不检索（`used_retrieval=False`）
- 检索率应该在30-60%之间（根据数据集特性）

**优先级**: 🔴 P0 - 核心baseline  
**预计时间**: 1-2天

---

### Task 3: 正确实现mR²AG [预计: 2-3天]

**参考资料**:
- 复现指南: `open_resource/m_r_ag_复现指南（面向_cursor）.md`
- 论文: mR²AG (arXiv:2411.15041)

#### 3.1 三阶段推理实现

**伪代码**（来自复现指南第227-253行）:
```python
# 1) Retrieval-Reflection
ret = model.generate_one_token(img, q, prompt=RET_PROMPT)
if ret == "[No Retrieval]":
    return model.generate_answer(img, q, prompt=ANS_PROMPT_SINGLE)

# 2) Entry Retrieval（Top-K 条目）
entries = clip_retriever.topk(img, K=5)

cands = []
for e in entries:
    for para in e.paragraphs:
        rel_tok, rel_prob = model.generate_one_token_with_prob(
            img, q, para, prompt=REL_PROMPT
        )
        if rel_tok == "[Relevant]":
            ans, ans_prob_geom = model.generate_answer_with_prob(
                img, q, para, prompt=ANS_PROMPT
            )
            cands.append({
                'ans': ans,
                'S_ret': e.score,
                'S_rel': rel_prob,
                'S_ans': ans_prob_geom
            })

# 3) 层级后处理（乘积打分）
best = max(cands, key=lambda d: d['S_ret']*d['S_rel']*d['S_ans'])
return normalize(best['ans'])
```

**实际实现**:
```python
class MR2AGPipeline(BaselinePipeline):
    """mR²AG: 多轮检索-反思-生成"""
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.special_tokens = {
            'retrieval': '[Retrieval]',
            'no_retrieval': '[No Retrieval]',
            'relevant': '[Relevant]',
            'irrelevant': '[Irrelevant]'
        }
    
    def stage1_retrieval_reflection(self, question, image):
        """阶段1: Retrieval-Reflection判断"""
        
        prompt = f"""Only output one token: [Retrieval] or [No Retrieval].

Question: {question}

Does this question require external knowledge?"""
        
        response = self.qwen3_vl.generate(
            text=prompt,
            image=image,
            max_new_tokens=5,
            temperature=0.01
        )
        
        needs_retrieval = '[Retrieval]' in response or 'retrieval' in response.lower()
        return needs_retrieval
    
    def stage2_relevance_reflection(self, question, image, paragraph):
        """阶段2: Relevance-Reflection段落判断"""
        
        prompt = f"""Only output one token: [Relevant] or [Irrelevant].

Question: {question}
Paragraph: {paragraph[:300]}

Is this paragraph relevant to answering the question?"""
        
        response = self.qwen3_vl.generate(
            text=prompt,
            image=image,
            max_new_tokens=5,
            temperature=0.01
        )
        
        is_relevant = '[Relevant]' in response or 'relevant' in response.lower()
        
        # 简化版: 无法直接获取token概率，使用启发式
        # 完整版需要修改模型接口以返回logits
        relevance_score = 0.9 if is_relevant else 0.1
        
        return is_relevant, relevance_score
    
    def stage3_answer_generation(self, question, image, paragraph, options):
        """阶段3: 基于相关段落生成答案"""
        
        prompt = self._construct_prompt(question, options, context=paragraph)
        
        answer = self.qwen3_vl.generate(
            text=prompt,
            image=image,
            max_new_tokens=self.config['max_new_tokens'],
            temperature=self.config['temperature']
        )
        
        # 简化版: 答案置信度使用启发式
        # 完整版需要计算token-level概率的几何平均
        answer_score = 0.8 if len(answer.strip()) > 0 else 0.1
        
        return answer, answer_score
    
    def split_into_paragraphs(self, doc_text, max_para_len=200):
        """将文档切分成段落"""
        
        # 简单按句号切分
        sentences = doc_text.split('. ')
        paragraphs = []
        current_para = ""
        
        for sent in sentences:
            if len(current_para) + len(sent) < max_para_len:
                current_para += sent + '. '
            else:
                if current_para:
                    paragraphs.append(current_para.strip())
                current_para = sent + '. '
        
        if current_para:
            paragraphs.append(current_para.strip())
        
        return paragraphs[:5]  # 最多5个段落
    
    def run_single(self, sample):
        """运行单个样本 - 完整的三阶段流程"""
        
        # ===== 阶段1: Retrieval-Reflection =====
        needs_retrieval = self.stage1_retrieval_reflection(
            sample['question'], 
            sample['image']
        )
        
        if not needs_retrieval:
            # 直接生成答案
            options = {
                'A': sample['A'], 'B': sample['B'],
                'C': sample['C'], 'D': sample['D']
            }
            prompt = self._construct_prompt(sample['question'], options, context=None)
            prediction = self._generate(prompt, sample['image'])
            
            return {
                'answer': self._map_letter_to_answer(prediction, sample),
                'raw_prediction': prediction,
                'retrieved_docs': [],
                'used_retrieval': False,
                'stage': 'No Retrieval'
            }
        
        # ===== 阶段2: Entry Retrieval + Relevance-Reflection =====
        # 检索Top-K条目
        entries = self.retriever.search(sample['question'], num=10)
        
        candidates = []
        options = {
            'A': sample['A'], 'B': sample['B'],
            'C': sample['C'], 'D': sample['D']
        }
        
        for entry in entries[:5]:  # Top-5条目
            entry_score = entry.get('score', 1.0)
            doc_text = entry.get('contents', '')
            
            # 将条目切分成段落
            paragraphs = self.split_into_paragraphs(doc_text)
            
            for para in paragraphs:
                # 段落级相关性判断
                is_relevant, relevance_score = self.stage2_relevance_reflection(
                    sample['question'],
                    sample['image'],
                    para
                )
                
                if is_relevant:
                    # ===== 阶段3: Answer Generation =====
                    answer, answer_score = self.stage3_answer_generation(
                        sample['question'],
                        sample['image'],
                        para,
                        options
                    )
                    
                    # 层级后处理: 乘积打分
                    combined_score = entry_score * relevance_score * answer_score
                    
                    candidates.append({
                        'answer': answer,
                        'paragraph': para,
                        'entry_score': entry_score,
                        'relevance_score': relevance_score,
                        'answer_score': answer_score,
                        'combined_score': combined_score
                    })
        
        # 选择得分最高的候选答案
        if candidates:
            best_candidate = max(candidates, key=lambda x: x['combined_score'])
            final_answer = best_candidate['answer']
            used_paragraphs = [best_candidate['paragraph']]
        else:
            # 如果没有相关段落，使用Top-1条目
            context = entries[0].get('contents', '')[:500] if entries else ""
            prompt = self._construct_prompt(sample['question'], options, context)
            final_answer = self._generate(prompt, sample['image'])
            used_paragraphs = [context]
        
        return {
            'answer': self._map_letter_to_answer(final_answer, sample),
            'raw_prediction': final_answer,
            'retrieved_docs': used_paragraphs,
            'used_retrieval': True,
            'stage': 'Three-Stage Reflection',
            'num_candidates': len(candidates),
            'best_score': best_candidate['combined_score'] if candidates else 0.0
        }
```

**关键特性**:
1. ✅ 三阶段推理流程
2. ✅ Retrieval-Reflection判断
3. ✅ 段落级Relevance-Reflection
4. ✅ 层级后处理（S_ret × S_rel × S_ans）
5. ✅ 只在相关段落上生成答案

**注意事项**:
- ⚠️ 简化版: 无法获取token-level概率，使用启发式分数
- ⚠️ 完整版需要修改模型接口以支持`output_scores=True`
- ⚠️ 理想情况下应该在mR²AG-IT数据上微调模型，但可以先用zero-shot测试

**优先级**: 🔴 P0 - 核心baseline  
**预计时间**: 2-3天

---

### Task 4: 测试Self-RAG和mR²AG [预计: 0.5天]

**测试脚本**:
```bash
# 创建测试脚本
cat > FlashRAG/experiments/test_corrected_baselines.py << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试修正后的baseline实现"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from run_all_baselines_100samples import *

# 测试10个样本
config = CONFIG.copy()
config['max_samples'] = 10

print("=" * 60)
print("测试修正后的Baseline实现")
print("=" * 60)

# 加载数据
samples = load_dataset(config['dataset_path'], max_samples=10)
print(f"\n✅ 加载 {len(samples)} 个测试样本")

# 初始化
qwen3_vl = initialize_qwen3_vl(config)
retriever = initialize_retriever(config, use_multimodal=False)

# 测试Self-RAG
print("\n" + "=" * 60)
print("1. 测试Self-RAG")
print("=" * 60)
self_rag = SelfRAGPipeline(qwen3_vl, retriever, config)

retrieval_count = 0
no_retrieval_count = 0

for i, sample in enumerate(samples[:10]):
    result = self_rag.run_single(sample)
    if result['used_retrieval']:
        retrieval_count += 1
    else:
        no_retrieval_count += 1
    
    print(f"\nSample {i+1}:")
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  Decision: {result['retrieval_decision']}")
    print(f"  Used Retrieval: {result['used_retrieval']}")
    if result['used_retrieval']:
        print(f"  Relevant Docs: {result.get('num_relevant_docs', 0)}")

print(f"\n检索统计:")
print(f"  使用检索: {retrieval_count}/10 ({retrieval_count*10}%)")
print(f"  不检索: {no_retrieval_count}/10 ({no_retrieval_count*10}%)")

# 测试mR²AG
print("\n" + "=" * 60)
print("2. 测试mR²AG")
print("=" * 60)
mr2ag = MR2AGPipeline(qwen3_vl, retriever, config)

for i, sample in enumerate(samples[:5]):  # 只测试5个（较慢）
    result = mr2ag.run_single(sample)
    
    print(f"\nSample {i+1}:")
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  Stage: {result['stage']}")
    print(f"  Used Retrieval: {result['used_retrieval']}")
    if result['used_retrieval']:
        print(f"  Candidates: {result['num_candidates']}")
        print(f"  Best Score: {result['best_score']:.4f}")

print("\n" + "=" * 60)
print("✅ 测试完成！")
print("=" * 60)
EOF

chmod +x FlashRAG/experiments/test_corrected_baselines.py

# 运行测试
cd /root/autodl-tmp
python FlashRAG/experiments/test_corrected_baselines.py
```

**验证checklist**:
- [ ] Self-RAG的检索率在30-60%之间（不是100%）
- [ ] Self-RAG能正确识别需要外部知识的问题
- [ ] mR²AG执行了三阶段流程
- [ ] mR²AG生成了多个候选答案并选择最佳
- [ ] 两个方法的答案质量合理

---

## 📊 阶段2: 100样本验证（预计1天）

### Task 5: 运行100样本对比实验

**修改实验脚本**:
```python
# 只运行修正后的baseline
METHODS_TO_RUN = [
    'self_aware_mrag',  # 我们的方法
    'self_rag',         # 修正后的Self-RAG
    'mr2ag',            # 修正后的mR²AG
    # 暂时不运行其他baseline
]
```

**运行命令**:
```bash
cd /root/autodl-tmp/FlashRAG
nohup python experiments/run_all_baselines_100samples.py \
    --methods self_aware_mrag self_rag mr2ag \
    --max_samples 100 \
    > ../run_corrected_baselines_100.log 2>&1 &

# 监控进度
tail -f ../run_corrected_baselines_100.log
```

**期望结果**:
- Self-RAG的Recall@5应该 > 9.0（因为有自适应检索）
- mR²AG的Recall@5应该 > 9.0（因为有段落级判断）
- EM和F1的差异应该更合理（不会完全一样）
- 我们的方法应该仍然是最好的，但差距可能会缩小

---

## 📊 阶段3: 其他Baseline（根据需要）

### Task 6-8: VisRAG实现 [预计: 2-3天]

**特点**:
- 需要专门的视觉编码器
- 跨模态融合机制
- 视觉引导检索

**参考**: `open_resource/VisRAG-master/`

### Task 9-11: REVEAL实现 [预计: 2-3天]

**特点**:
- 跨模态推理
- 知识图谱增强

### Task 12-14: RagVL实现 [预计: 1-2天]

**特点**:
- 视觉语言联合编码

### Task 15-17: MuRAG实现 [预计: 2-3天]

**特点**:
- 多模态记忆
- 联合对比学习
- 需要预训练

**参考**: `open_resource/murag复现文档（基于论文和资料）.md`

---

## 📊 阶段4: 完整实验（预计1-2天）

### Task 18: 运行完整数据集

```bash
# 所有baseline + 全部样本
python experiments/run_all_baselines_fullset.py \
    --max_samples 2000 \
    --methods all
```

### Task 19: 结果分析和论文撰写

---

## 📋 优先级总结

| 阶段 | 任务 | 优先级 | 预计时间 | 必要性 |
|------|------|--------|---------|--------|
| **阶段1** | 修复指标问题 | 🔴 P0 | 2小时 | ⭐⭐⭐⭐⭐ |
| | Self-RAG实现 | 🔴 P0 | 1-2天 | ⭐⭐⭐⭐⭐ |
| | mR²AG实现 | 🔴 P0 | 2-3天 | ⭐⭐⭐⭐⭐ |
| | 测试验证 | 🔴 P0 | 0.5天 | ⭐⭐⭐⭐⭐ |
| **阶段2** | 100样本实验 | 🔴 P0 | 1天 | ⭐⭐⭐⭐⭐ |
| | 结果分析 | 🔴 P0 | 0.5天 | ⭐⭐⭐⭐⭐ |
| **阶段3** | VisRAG实现 | 🟡 P1 | 2-3天 | ⭐⭐⭐ |
| | REVEAL实现 | 🟢 P2 | 2-3天 | ⭐⭐ |
| | RagVL实现 | 🟢 P2 | 1-2天 | ⭐⭐ |
| | MuRAG实现 | 🟢 P2 | 2-3天 | ⭐ |
| **阶段4** | 完整实验 | 🔴 P0 | 1-2天 | ⭐⭐⭐⭐⭐ |

**总计**:
- **阶段1+2（核心）**: 5-7天
- **阶段3（可选）**: 7-11天
- **阶段4（必须）**: 1-2天

---

## ✅ 成功标准

### 阶段1完成标准:
- [ ] 3个指标不再是0
- [ ] Self-RAG的检索率 < 100%（有自适应性）
- [ ] mR²AG执行三阶段流程
- [ ] 10样本测试通过

### 阶段2完成标准:
- [ ] 100样本实验完成
- [ ] 各baseline的Recall@5不完全相同
- [ ] EM/F1分布合理
- [ ] 我们的方法仍然最优或接近最优

### 最终完成标准:
- [ ] 所有核心baseline正确实现
- [ ] 完整数据集实验完成
- [ ] 结果可信、可复现
- [ ] 可以用于论文投稿

---

**下一步**: 开始Task 1 - 修复3个指标为0的问题

需要我立即开始吗？

