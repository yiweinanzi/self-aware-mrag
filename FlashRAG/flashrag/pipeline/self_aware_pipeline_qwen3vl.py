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
                mllm_model=None,
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
        
        # ========== 阶段1: 不确定性估计 ==========
        uncertainty = self.uncertainty_estimator.estimate(question, image)
        
        if isinstance(uncertainty, dict):
            total_unc = uncertainty.get('total', 0.5)
            uncertainty_info = uncertainty
        else:
            total_unc = uncertainty
            uncertainty_info = {'total': total_unc}
        
        # 🔍 DEBUG: 输出不确定性值
        print(f"[DEBUG] uncertainty={total_unc:.4f}, threshold={self.uncertainty_threshold:.4f}, should_retrieve={total_unc > self.uncertainty_threshold}")
        
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
            
            # 位置感知融合
            position_bias_stats = None
            if self.use_position_fusion and retrieved_docs:
                fused_docs, fused_scores, position_bias_stats = self._apply_position_fusion(
                    retrieved_docs, retrieval_scores, question
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
        
        # ✅ 使用Qwen3-VL生成
        text_answer = self._generate_answer_qwen3vl(question, context, image)
        
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
            'golden_answers': golden_answers
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
                               query: str) -> Tuple[List[str], List[float], Dict]:
        """
        应用位置感知融合
        
        Returns:
            fused_docs: 融合后的文档
            fused_scores: 融合后的分数
            position_bias_stats: 位置偏差统计信息
        """
        if not docs:
            return [], [], None
        
        k = len(docs)
        
        # 计算位置权重
        position_weights = np.exp(np.arange(k) * 0.5)
        position_weights = position_weights / position_weights.sum()
        
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
            'original_scores': scores,
            'combined_scores': combined_weights.tolist(),
            'reordering_magnitude': float(np.mean(np.abs(np.array(sorted_indices) - np.arange(k)))),
            'top1_changed': int(sorted_indices[0] != 0) if len(sorted_indices) > 0 else 0,
        }
        
        return reordered_docs[:3], reordered_scores[:3], position_bias_stats
    
    def _format_context_with_attribution_preview(self, docs: List[str], 
                                                  scores: List[float],
                                                  attributions: Optional[List] = None) -> str:
        """使用归因信息格式化上下文"""
        context_parts = []
        
        for i, (doc, score) in enumerate(zip(docs, scores)):
            if attributions and i < len(attributions):
                attr = attributions[i]
                attr_conf = attr.get('confidence', 0) if isinstance(attr, dict) else 0
                combined_score = (score + attr_conf) / 2
            else:
                combined_score = score
            
            if combined_score > 0.6:
                importance = "**HIGH RELEVANCE**"
            elif combined_score > 0.3:
                importance = "**RELEVANT**"
            else:
                importance = "**REFERENCE**"
            
            doc_text = doc[:300] if len(doc) > 300 else doc
            citation = f" [Confidence: {combined_score:.2f}]" if attributions else ""
            
            context_parts.append(
                f"[Evidence {i+1}] {importance}{citation}\n{doc_text}"
            )
        
        return "\n\n".join(context_parts)
    
    def _generate_answer_qwen3vl(self, question: str, context: str, image=None) -> str:
        """
        ✅ 使用Qwen3-VL生成答案
        
        支持：
        - 单图像生成
        - 多图像生成（如果context包含多图像）
        - 高分辨率图像
        """
        # 构建prompt
        if context:
            prompt = f"""Based on the following evidence, answer the question concisely.

{context}

Question: {question}

Answer:"""
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

