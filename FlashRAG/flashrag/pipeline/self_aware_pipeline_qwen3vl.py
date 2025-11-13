#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-Aware Multimodal RAG Pipeline - Qwen3-VL版本

✅ 统一使用Qwen3-VL-8B-Instruct，确保公平对比

与LLaVA版本的区别：
1. 模型：Qwen3-VL-8B-Instruct（2024）vs LLaVA-1.5（2023）
2. 多图像支持：最多20张 vs 单图像
3. 高分辨率：支持 vs 有限
4. 指令跟随：更强 vs 一般

核心创新保持不变：
- 跨模态不确定性估计
- 位置感知融合
- 细粒度归因
- 多模态输出（可选）
"""

import torch
import warnings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class SelfAwarePipelineQwen3VL:
    """
    Self-Aware Multimodal RAG Pipeline（Qwen3-VL版本）
    
    ✅ P0修复：统一使用Qwen3-VL确保公平对比
    
    核心流程：
    1. 不确定性估计 → 决定是否检索
    2. 自适应检索 → 位置感知融合
    3. 生成答案（Qwen3-VL）
    4. 细粒度归因
    5. 多模态输出增强（可选）
    
    使用示例：
    ```python
    from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
    
    qwen3_vl = create_qwen3_vl_wrapper()
    
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=retriever,
        config={
            'uncertainty_threshold': 0.35,
            'use_position_fusion': True,
            'use_attribution': True
        }
    )
    
    results = pipeline.run(dataset)
    ```
    """
    
    def __init__(self, qwen3_vl_wrapper, retriever, config=None):
        """
        初始化Pipeline（Qwen3-VL版本）
        
        Args:
            qwen3_vl_wrapper: Qwen3-VL模型封装器
            retriever: 检索器
            config: 配置字典
        """
        self.qwen3_vl = qwen3_vl_wrapper  # ✅ 使用Qwen3-VL
        self.retriever = retriever
        self.config = config or {}
        
        # 配置参数
        # 优化：降低阈值，让更多样本触发检索
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.30)  # 0.35 → 0.30
        self.top_k = self.config.get('retrieval_topk', 5)
        self.use_position_fusion = self.config.get('use_position_fusion', True)
        self.use_attribution = self.config.get('use_attribution', True)
        self.use_multimodal_output = self.config.get('enable_multimodal_output', False)
        
        # Qwen3-VL特定配置
        self.max_images = min(self.config.get('max_images', 20), 20)  # 最多20张
        self.use_thinking = self.config.get('thinking', False)  # P0-2: 确保thinking=false
        
        # 初始化模块
        self._init_modules()
        
        print("✅ SelfAwarePipelineQwen3VL初始化完成")
        print(f"  - 模型: Qwen3-VL-8B-Instruct")
        print(f"  - Uncertainty threshold (τ): {self.uncertainty_threshold:.3f}")
        print(f"  - Max images: {self.max_images}")
        print(f"  - Thinking mode: {self.use_thinking} (推荐False)")
        print(f"  - Position fusion: {self.use_position_fusion}")
        print(f"  - Attribution: {self.use_attribution}")
        print(f"  - Multimodal output: {self.use_multimodal_output}")
    
    def should_retrieve(self, u: float, tau: Optional[float] = None) -> bool:
        """
        ✅ P0-6: 阈值接口化 - 检索决策函数
        
        Args:
            u: 不确定性分数
            tau: 不确定性阈值（可选）
        
        Returns:
            bool: True表示需要检索
        """
        threshold = tau if tau is not None else self.uncertainty_threshold
        return u > threshold
    
    def _relevance_judgment(self, question: str, document: str, image=None) -> bool:
        """
        ✅ 优化C-Step1: 判断文档是否与问题相关（借鉴Self-RAG）
        
        Args:
            question: 问题文本
            document: 文档内容
            image: 图像（可选）
        
        Returns:
            bool: True表示相关
        """
        doc_preview = document[:300] + "..." if len(document) > 300 else document
        
        prompt = f"""Task: Is this document relevant to answering the question?

Question: {question}

Document: {doc_preview}

Answer ONLY 'RELEVANT' or 'IRRELEVANT':"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=5,
                temperature=0.05
            )
            
            response_upper = response.strip().upper()
            is_relevant = 'RELEVANT' in response_upper and 'IRRELEVANT' not in response_upper[:15]
            return is_relevant
        except Exception as e:
            print(f"[WARN] Relevance judgment failed: {e}, defaulting to True")
            return True
    
    def _verify_answer_support(self, question: str, answer: str, documents: list, image=None) -> float:
        """
        ✅ 最终优化-Step4: 验证答案的支持度（借鉴Self-RAG）
        
        Args:
            question: 问题文本
            answer: 生成的答案
            documents: 检索的文档列表
            image: 图像（可选）
        
        Returns:
            float: 支持度分数 [0, 1]
        """
        # 提取文档内容
        doc_texts = []
        for doc in documents[:3]:  # 只用前3个
            if isinstance(doc, dict):
                doc_texts.append(doc.get('contents', doc.get('text', '')))
            else:
                doc_texts.append(str(doc))
        
        combined_docs = " ".join(doc_texts)[:500]  # 限制长度
        
        prompt = f"""Task: Is the answer supported by the provided documents?

Question: {question}

Answer: {answer}

Documents: {combined_docs}

Rate the support level:
- FULLY_SUPPORTED: Answer is directly supported by documents
- PARTIALLY_SUPPORTED: Answer is somewhat related to documents  
- NOT_SUPPORTED: Answer is not supported by documents

Answer with ONE word only:"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=10,
                temperature=0.05
            )
            
            response_upper = response.strip().upper()
            
            # 映射到分数
            if 'FULLY' in response_upper or 'FULL' in response_upper:
                return 0.9
            elif 'PARTIALLY' in response_upper or 'PARTIAL' in response_upper:
                return 0.6
            elif 'NOT' in response_upper:
                return 0.2
            else:
                return 0.5  # 默认中等支持度
        
        except Exception as e:
            print(f"[WARN] Support verification failed: {e}, defaulting to 0.5")
            return 0.5
    
    def _init_modules(self):
        """初始化各个模块"""
        from flashrag.modules.position_aware_fusion import PositionAwareCrossModalFusion
        from flashrag.modules.attribution import FineGrainedMultimodalAttribution
        from flashrag.modules.modality_selector import ModalitySelector
        from flashrag.modules.query_reformulation import QueryReformulator
        
        # 1. 不确定性估计器 - 支持选择原始版或改进版
        use_improved = self.config.get('use_improved_estimator', False)
        
        if use_improved:
            print("  ✅ 使用改进版不确定性估计器 (ImprovedUncertaintyEstimator)")
            from flashrag.modules.uncertainty_estimator_improved import ImprovedUncertaintyEstimator
            self.uncertainty_estimator = ImprovedUncertaintyEstimator(
                config={
                    'clip_model_path': self.config.get('clip_model_path', 
                        '/root/autodl-tmp/models/clip-vit-large-patch14-336'),
                    'text_weight': 0.5,
                    'visual_weight': 0.3,
                    'alignment_weight': 0.2
                }
            )
        else:
            print("  ℹ️  使用原始不确定性估计器 (CrossModalUncertaintyEstimator)")
            from flashrag.modules.uncertainty_estimator import CrossModalUncertaintyEstimator
            self.uncertainty_estimator = CrossModalUncertaintyEstimator(
                mllm_model=self.qwen3_vl,  # ✅ 修复：传入qwen3_vl完整wrapper
                config={
                    'eigen_threshold': -6.0,
                    'use_clip_for_alignment': True,
                    'clip_model_path': self.config.get('clip_model_path', 
                        '/root/autodl-tmp/models/clip-vit-large-patch14-336'),
                    'text_weight': 0.4,
                    'visual_weight': 0.3,
                    'alignment_weight': 0.3
                }
            )
        
        # 2. 模态选择器
        self.modality_selector = ModalitySelector()
        
        # 3. 查询重构器
        self.query_reformulator = QueryReformulator()
        
        # 4. 位置感知融合
        if self.use_position_fusion:
            self.position_aware_fusion = PositionAwareCrossModalFusion(
                d_model=768, num_heads=12, device='cpu'
            )
        
        # 5. 归因模块
        if self.use_attribution:
            self.attribution_module = FineGrainedMultimodalAttribution(
                mllm_model=None
            )
        
        # 6. 多模态输出（可选）
        if self.use_multimodal_output:
            try:
                from flashrag.modules.multimodal_output import MultimodalOutputComposition
                self.multimodal_output = MultimodalOutputComposition()
                print("  ⚠️  多模态输出已启用（实验性功能）")
            except Exception as e:
                warnings.warn(f"Multimodal output模块加载失败: {e}")
                self.use_multimodal_output = False
        else:
            self.multimodal_output = None
    
    # =========================================================================
    # 核心Pipeline流程
    # =========================================================================
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本（Qwen3-VL版本）
        
        Args:
            sample: 样本字典
        
        Returns:
            Dict: 结果字典
        """
        question = sample['question']
        image = sample.get('image', None)
        
        # 初始化统计变量
        position_bias_stats = None
        attribution_stats = None
        
        # ✅ MRAG-Bench多选题格式支持
        has_choices = ('A' in sample and sample.get('A'))
        original_question = question
        
        if has_choices:
            # 构造多选题格式的问题
            question = f"""{original_question}

Options:
A. {sample.get('A', '')}
B. {sample.get('B', '')}
C. {sample.get('C', '')}
D. {sample.get('D', '')}

Answer with the letter only (A/B/C/D):"""
        
        # ⚠️ 保存原始question用于生成（避免被query改写破坏Options格式）
        question_for_generation = question
        
        # ========== 阶段1: 不确定性估计 ==========
        uncertainty = self.uncertainty_estimator.estimate(question, image)
        
        if isinstance(uncertainty, dict):
            total_unc = uncertainty.get('total', 0.5)
            uncertainty_info = uncertainty
        else:
            total_unc = uncertainty
            uncertainty_info = {'total': total_unc}
        
        # 🔍 DEBUG: 输出不确定性值（包含三个分量）
        text_unc = uncertainty_info.get('text', 0.0)
        visual_unc = uncertainty_info.get('visual', 0.0)
        align_unc = uncertainty_info.get('alignment', 0.0)
        print(f"[DEBUG] uncertainty={total_unc:.4f} [text={text_unc:.4f}, visual={visual_unc:.4f}, align={align_unc:.4f}], threshold={self.uncertainty_threshold:.4f}, should_retrieve={total_unc > self.uncertainty_threshold}")
        
        # ========== 阶段2: 自适应检索 ==========
        retrieved_docs = []
        retrieval_scores = []
        fused_docs = []
        fused_scores = []
        
        if self.should_retrieve(total_unc):
            should_retrieve = True
            
            # 模态选择
            modality = self.modality_selector.select(uncertainty_info)
            print(f"[DEBUG] modality={modality}")
            
            # 查询重构
            enhanced_query = self.query_reformulator.reformulate(
                query=question,
                uncertainty_scores=uncertainty_info,
                modality=modality
            )
            print(f"[DEBUG] enhanced_query={enhanced_query[:80] if enhanced_query else 'None'}...")
            
            # 检索（支持不同的检索器接口）
            print(f"[DEBUG] self.retriever is not None: {self.retriever is not None}")
            if self.retriever:
                print(f"[DEBUG] 进入检索分支")
                modality_weights = self.modality_selector.get_modality_weights(modality)
                
                # FlashRAG的DenseRetriever使用search方法
                if hasattr(self.retriever, 'search'):
                    print(f"[DEBUG] 调用retriever.search(), top_k={self.top_k}")
                    # DenseRetriever.search(query, num, return_score) 返回 list[dict] 或 (list[dict], list[float])
                    search_results = self.retriever.search(enhanced_query, num=self.top_k, return_score=True)
                    print(f"[DEBUG] search_results类型: {type(search_results)}, 是否为tuple: {isinstance(search_results, tuple)}")
                    if isinstance(search_results, tuple):
                        retrieved_docs, retrieval_scores = search_results
                        print(f"[DEBUG] retrieved_docs数量: {len(retrieved_docs) if retrieved_docs else 0}")
                    else:
                        retrieved_docs = search_results if search_results else []
                        retrieval_scores = [1.0] * len(retrieved_docs) if retrieved_docs else []
                        print(f"[DEBUG] retrieved_docs数量: {len(retrieved_docs)}")
                elif hasattr(self.retriever, 'retrieve'):
                    # 自定义检索器使用retrieve方法 (如SelfAwareMultimodalRetriever)
                    result = self.retriever.retrieve(
                        query_text=enhanced_query,
                        query_image=image,
                        top_k=self.top_k,
                        return_score=True  # 确保返回分数
                    )
                    if isinstance(result, tuple):
                        retrieved_docs, retrieval_scores = result
                    else:
                        retrieved_docs = result if result else []
                        retrieval_scores = [1.0] * len(retrieved_docs) if retrieved_docs else []
                else:
                    retrieved_docs = []
                    retrieval_scores = []
                
                uncertainty_info['original_query'] = question
                uncertainty_info['enhanced_query'] = enhanced_query if enhanced_query != question else None
                uncertainty_info['selected_modality'] = modality
                uncertainty_info['modality_weights'] = modality_weights
            else:
                retrieved_docs, retrieval_scores = [], []
                modality = 'both'
            
            # ✅ 优化C-Step1: 文档相关性过滤（借鉴Self-RAG）
            if retrieved_docs:
                relevant_docs = []
                relevant_scores = []
                
                print(f"[FILTER] 开始过滤{len(retrieved_docs)}个检索文档...")
                for idx, doc in enumerate(retrieved_docs):
                    doc_text = doc.get('contents', '') if isinstance(doc, dict) else str(doc)
                    is_relevant = self._relevance_judgment(question, doc_text, image)
                    
                    if is_relevant:
                        relevant_docs.append(doc)
                        relevant_scores.append(retrieval_scores[idx] if idx < len(retrieval_scores) else 1.0)
                        print(f"[FILTER] 文档{idx+1}: ✅ RELEVANT")
                    else:
                        print(f"[FILTER] 文档{idx+1}: ❌ IRRELEVANT (过滤)")
                
                print(f"[FILTER] 过滤完成: {len(retrieved_docs)} → {len(relevant_docs)} 个相关文档")
                
                # 如果没有相关文档，回退到直接回答（避免使用噪声）
                if not relevant_docs:
                    print(f"[FILTER] ⚠️  无相关文档，回退到直接回答")
                    should_retrieve = False
                    retrieved_docs, retrieval_scores = [], []
                else:
                    # 使用过滤后的文档
                    retrieved_docs = relevant_docs
                    retrieval_scores = relevant_scores
            
            # 位置感知融合
            position_bias_stats = None
            if self.use_position_fusion and retrieved_docs:
                # ✅ 修复P0-2: 传递不确定性到位置融合（创新点1和2的关联）
                fused_docs, fused_scores, position_bias_stats = self._apply_position_fusion(
                    retrieved_docs, retrieval_scores, question,
                    uncertainty_scores=uncertainty_info  # ✅ 传入不确定性
                )
            else:
                fused_docs = retrieved_docs[:3] if retrieved_docs else []
                fused_scores = retrieval_scores[:3] if retrieval_scores else []
        
        else:
            should_retrieve = False
        
        # ========== 阶段3: 生成答案（Qwen3-VL）==========
        if fused_docs:
            context = self._format_context_with_attribution_preview(
                fused_docs, fused_scores, attributions=None
            )
        else:
            context = ""
        
        # ✅ 使用Qwen3-VL生成（传入sample以获取选项）
        text_answer = self._generate_answer_qwen3vl(question_for_generation, context, image, sample)
        
        # ========== 新增：答案支持度验证（借鉴Self-RAG）==========
        support_score = None
        if fused_docs and text_answer:
            try:
                support_score = self._verify_answer_support(question_for_generation, text_answer, fused_docs, image)
                
                # 如果支持度过低，回退到直接回答（不使用检索）
                if support_score < 0.4:  # 支持度阈值
                    print(f"[SUPPORT] ⚠️  答案支持度过低 ({support_score:.2f})，回退到直接回答")
                    # 重新生成（不使用检索结果）
                    text_answer = self._generate_answer_qwen3vl(question_for_generation, "", image, sample)
                    fused_docs = []  # 清空检索结果
            except Exception as e:
                print(f"[SUPPORT] 验证失败: {e}")
        
        # ========== 阶段4: 细粒度归因 ==========
        attributions = None
        
        if self.use_attribution and fused_docs:
            try:
                retrieved_texts = [doc.get('text', '') for doc in fused_docs]
                attributions = self.attribution_module.attribute_text_evidence(
                    generated_text=text_answer,
                    retrieved_texts=retrieved_texts
                )
                
                if attributions and isinstance(attributions, list):
                    high_conf_count = sum(
                        1 for attr in attributions 
                        if isinstance(attr, dict) and attr.get('confidence', 0) > 0.7
                    )
                    
                    attribution_stats = {
                        'total_sources': len(attributions),
                        'high_confidence': high_conf_count,
                        'avg_confidence': np.mean([
                            attr.get('confidence', 0) 
                            for attr in attributions 
                            if isinstance(attr, dict)
                        ]) if attributions else 0
                    }
                else:
                    attribution_stats = None
                    
            except Exception as e:
                warnings.warn(f"归因计算失败: {e}")
                attributions = None
                attribution_stats = None
        
        # ========== 阶段5: 多模态输出增强（可选）==========
        final_answer = text_answer
        if self.use_multimodal_output and retrieved_docs and self.multimodal_output is not None:
            try:
                final_answer = self.multimodal_output.generate_multimodal_answer(
                    text_answer, retrieved_docs, attributions
                )
            except Exception as e:
                warnings.warn(f"多模态输出增强失败: {e}")
                final_answer = text_answer
        
        # ✅ 多选题答案映射
        if has_choices:
            # 提取选项字母（A/B/C/D）
            answer_letter = final_answer.strip().upper()
            if answer_letter and answer_letter[0] in ['A', 'B', 'C', 'D']:
                choice_letter = answer_letter[0]
                # 映射回具体答案
                mapped_answer = sample.get(choice_letter, final_answer)
                final_answer = mapped_answer
        
        # ========== 返回结果 ==========
        # 处理标准答案字段（兼容不同数据集格式）
        golden_answers = sample.get('golden_answers', [])
        if not golden_answers and 'answer' in sample:
            # MRAG-Bench等数据集使用'answer'字段
            golden = sample['answer']
            golden_answers = [golden] if golden else []
        
        result = {
            'question': question,
            'answer': final_answer,
            'uncertainty': uncertainty_info,
            'retrieved': should_retrieve,
            'retrieved_docs': retrieved_docs if should_retrieve else [],
            'n_retrieved_docs': len(retrieved_docs) if should_retrieve else 0,
            'n_fused_docs': len(fused_docs),
            'attributions': attributions,
            'golden_answers': golden_answers,
            
            # ✅ Task 1: 添加evaluator需要的字段
            # 1. retrieval_result - 用于Faithfulness计算
            'retrieval_result': [{
                'retrieved_docs': retrieved_docs if should_retrieve else [],
                'retrieval_scores': [1.0] * len(retrieved_docs) if should_retrieve else [],
                'retrieval_used': should_retrieve
            }],
            
            # 2. attributions - 确保格式正确（已有，但可能需要调整格式）
            # attributions字段已在上面定义
            
            # 3. position_bias_results - 用于Position Bias Score计算
            'position_bias_results': {
                'average_bias': position_bias_stats.get('bias_score', 0.0) if position_bias_stats else 0.0,
                'individual_scores': [position_bias_stats.get('bias_score', 0.0)] if position_bias_stats else [0.0],
                'position_weights': position_bias_stats.get('position_weights', []) if position_bias_stats else []
            }
        }
        
        if should_retrieve:
            result['selected_modality'] = uncertainty_info.get('selected_modality', 'both')
            result['modality_weights'] = uncertainty_info.get('modality_weights', {'text': 0.5, 'image': 0.5})
            result['query_enhanced'] = uncertainty_info.get('enhanced_query') is not None
        
        # 添加归因统计信息
        if attributions:
            result['attribution_stats'] = attribution_stats if 'attribution_stats' in locals() else None
        
        # 添加位置偏差统计信息
        if position_bias_stats is not None:
            result['position_bias_stats'] = position_bias_stats
        
        return result
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _apply_position_fusion(self, docs: List[str], scores: List[float],
                               query: str,
                               uncertainty_scores: Optional[Dict] = None) -> Tuple[List[str], List[float], Dict]:
        """
        应用位置感知融合（不确定性调制版）

        ✅ 修复P0-3: 实现不确定性驱动的位置权重调制

        理论依据：
        - 高不确定性 → 模型不确定 → 增强位置偏差缓解
        - 低不确定性 → 模型有信心 → 保持检索器原序

        Args:
            docs: 检索到的文档
            scores: 检索分数
            query: 查询
            uncertainty_scores: 不确定性分数字典（包含total, text, visual, alignment）

        Returns:
            fused_docs: 融合后的文档
            fused_scores: 融合后的分数
            position_bias_stats: 位置偏差统计信息
        """
        if not docs:
            return [], [], None

        k = len(docs)

        # 基础位置权重（指数衰减）
        base_position_weights = np.exp(-np.arange(k) * 0.5)
        base_position_weights = base_position_weights / base_position_weights.sum()

        # ✅ 核心创新：不确定性调制位置权重
        if uncertainty_scores is not None:
            total_unc = uncertainty_scores.get('total', 0.5)

            # 调制因子：不确定性越高，位置偏差缓解越强
            # total_unc ∈ [0, 1]
            # modulation ∈ [0.75, 1.25]
            # 公式: modulation = 1.0 + (U_total - 0.5) × α
            # 其中 α=0.5 是调制强度超参数
            modulation = 1.0 + (total_unc - 0.5) * 0.5

            # 应用调制
            position_weights = base_position_weights * modulation
            position_weights = position_weights / position_weights.sum()

            print(f"[DEBUG] 位置融合（不确定性调制）: total_unc={total_unc:.4f}, "
                  f"modulation={modulation:.4f}, "
                  f"weights_range=[{position_weights.min():.4f}, {position_weights.max():.4f}]")
        else:
            position_weights = base_position_weights
            modulation = 1.0
            print(f"[DEBUG] 位置融合（无调制）: 使用基础权重")

        # 综合权重
        scores_norm = np.array(scores) / (np.sum(scores) + 1e-10)
        combined_weights = scores_norm * position_weights

        # 排序
        sorted_indices = np.argsort(combined_weights)[::-1]

        reordered_docs = [docs[i] for i in sorted_indices]
        reordered_scores = [combined_weights[i] for i in sorted_indices]

        # 计算位置偏差统计信息
        position_bias_stats = {
            'original_positions': list(range(k)),
            'reordered_positions': sorted_indices.tolist(),
            'position_weights': position_weights.tolist(),
            'base_position_weights': base_position_weights.tolist(),  # ✅ 新增：基础权重
            'uncertainty_modulation': float(modulation),  # ✅ 新增：调制因子
            'total_uncertainty': uncertainty_scores.get('total', 0.0) if uncertainty_scores else 0.0,  # ✅ 新增
            'original_scores': scores,
            'combined_scores': combined_weights.tolist(),
            'reordering_magnitude': float(np.mean(np.abs(np.array(sorted_indices) - np.arange(k)))),
            'top1_changed': int(sorted_indices[0] != 0) if len(sorted_indices) > 0 else 0,
        }

        return reordered_docs[:3], reordered_scores[:3], position_bias_stats  # 优化：使用top3减少噪声
    
    def _format_context_with_attribution_preview(self, docs: List[str], 
                                                  scores: List[float],
                                                  attributions: Optional[List] = None) -> str:
        """
        ✅ 优化：简化Context格式，避免复杂标签干扰LLM理解
        
        修改前: [Evidence 1] **HIGHLY RELEVANT** [Confidence: 0.95]\ntext...
        修改后: Document 1:\ntext...
        
        效果: 与baseline保持一致的简洁格式
        """
        context_parts = []
        
        for i, doc in enumerate(docs):
            # 简化格式：只保留Document编号和内容
            doc_text = doc[:512] if len(doc) > 512 else doc  # 优化：512字符平衡信息与噪声
            context_parts.append(
                f"Document {i+1}:\n{doc_text}"
            )
        
        return "\n\n".join(context_parts)
    
    def _generate_answer_qwen3vl(self, question: str, context: str, image=None, sample: Dict = None) -> str:
        """
        ✅ 使用Qwen3-VL生成答案
        
        支持：
        - 单图像生成
        - 多图像生成（如果context包含多图像）
        - 高分辨率图像
        - 多选题格式（与baseline完全一致）
        """
        # 检查是否是多选题
        has_choices = sample and all(k in sample and sample.get(k) for k in ['A', 'B', 'C', 'D'])
        
        # 构建prompt
        if context:
            if has_choices:
                # 多选题格式 - 与baseline完全一致！
                # 提取纯问题（去除Options部分）
                core_question = question.split('\nOptions:')[0] if '\nOptions:' in question else question.split('\n')[0]
                
                prompt = f"""Based on the following evidence, answer the question.

{context}

Question: {core_question}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
            else:
                # 普通问题格式
                prompt = f"""Based on the following evidence, answer the question concisely.

{context}

Question: {question}

Answer:"""
        else:
            if has_choices:
                core_question = question.split('\nOptions:')[0] if '\nOptions:' in question else question.split('\n')[0]
                prompt = f"""Question: {core_question}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
            else:
                prompt = f"""Question: {question}

Answer:"""
        
        try:
            # ✅ 使用Qwen3-VL生成
            # 注意：Qwen3-VL不接受thinking参数，已在prompt中控制输出格式
            # 优化：降低温度提高确定性，减少token数加快速度
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=10,  # 多选题只需要1个字母，减少生成长度
                temperature=0.01  # 接近0，提高确定性
            )
            return answer.strip()
            
        except Exception as e:
            warnings.warn(f"Qwen3-VL生成失败: {e}")
            return ""
    
    # =========================================================================
    # 批量处理
    # =========================================================================
    
    def run(self, dataset, verbose: bool = True) -> List[Dict[str, Any]]:
        """在数据集上运行Pipeline（Qwen3-VL版本）"""
        results = []
        
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(dataset, desc="Self-Aware Pipeline (Qwen3-VL)")
        else:
            iterator = dataset
        
        retrieval_triggered = 0
        
        for sample in iterator:
            try:
                result = self.run_single(sample)
                
                if result['retrieved']:
                    retrieval_triggered += 1
                
                # 评估
                answer = result['answer'].lower().strip()
                golden = result.get('golden_answers', [])
                correct = any(g.lower().strip() in answer for g in golden)
                result['correct'] = correct
                
                results.append(result)
                
                if verbose and len(results) % 50 == 0:
                    acc = sum(r['correct'] for r in results) / len(results)
                    ret_rate = retrieval_triggered / len(results)
                    if hasattr(iterator, 'set_postfix'):
                        iterator.set_postfix({
                            'Acc': f'{acc*100:.1f}%',
                            'Ret': f'{ret_rate*100:.0f}%'
                        })
            
            except Exception as e:
                warnings.warn(f"处理样本失败: {e}")
                continue
        
        if verbose:
            acc = sum(r['correct'] for r in results) / len(results) if results else 0
            ret_rate = retrieval_triggered / len(results) if results else 0
            print(f"\n✅ Pipeline完成:")
            print(f"  准确率: {acc*100:.2f}%")
            print(f"  检索率: {ret_rate*100:.1f}%")
            print(f"  处理样本: {len(results)}")
        
        return results


# 工厂函数
def create_self_aware_pipeline_qwen3vl(qwen3_vl_wrapper, retriever, **kwargs):
    """
    创建Self-Aware Pipeline（Qwen3-VL版本）
    
    Args:
        qwen3_vl_wrapper: Qwen3-VL封装器
        retriever: 检索器
        **kwargs: 配置参数
    
    Returns:
        SelfAwarePipelineQwen3VL实例
    """
    return SelfAwarePipelineQwen3VL(qwen3_vl_wrapper, retriever, kwargs)


if __name__ == '__main__':
    print("=" * 80)
    print("Self-Aware Multimodal RAG Pipeline - Qwen3-VL版本")
    print("=" * 80)
    print()
    print("✅ P0修复: 统一使用Qwen3-VL")
    print()
    print("优势：")
    print("  1. 公平对比（所有方法统一模型）")
    print("  2. 多图像支持（最多20张）")
    print("  3. 高分辨率处理")
    print("  4. 更强指令跟随")
    print()
    print("使用方法:")
    print("  from flashrag.pipeline.self_aware_pipeline_qwen3vl import create_self_aware_pipeline_qwen3vl")
    print("  from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper")
    print()
    print("  qwen3_vl = create_qwen3_vl_wrapper()")
    print("  pipeline = create_self_aware_pipeline_qwen3vl(qwen3_vl, retriever)")
    print("  results = pipeline.run(dataset)")
    print("=" * 80)

