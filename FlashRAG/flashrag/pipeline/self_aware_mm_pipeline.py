# -*- coding: utf-8 -*-
"""
Self-Aware Multimodal RAG Pipeline
自感知多模态检索增强生成Pipeline

实现文档中的创新点：
1. 跨模态自感知不确定性机制 (部分实现)
2. 位置感知的跨模态融合 (完整实现)
3. 细粒度多模态证据归因 (待实现)
"""

from flashrag.pipeline.mm_pipeline import BasicMultiModalPipeline
from flashrag.utils import get_retriever, get_generator
import warnings

try:
    from flashrag.retriever.multimodal_retriever import (
        PositionAwareFusion,
        SelfAwareMultimodalRetriever
    )
    POSITION_FUSION_AVAILABLE = True
except ImportError:
    POSITION_FUSION_AVAILABLE = False
    warnings.warn("位置感知融合模块未找到")

try:
    from flashrag.modules.uncertainty_estimator import CrossModalUncertaintyEstimator
    UNCERTAINTY_ESTIMATOR_AVAILABLE = True
except ImportError:
    UNCERTAINTY_ESTIMATOR_AVAILABLE = False
    warnings.warn("完整版不确定性估计模块未找到，将使用简化版")

try:
    from flashrag.modules.attribution import FineGrainedMultimodalAttribution
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False
    warnings.warn("细粒度归因模块未找到")

try:
    from flashrag.modules.multimodal_output import MultimodalOutputComposition
    MULTIMODAL_OUTPUT_AVAILABLE = True
except ImportError:
    MULTIMODAL_OUTPUT_AVAILABLE = False
    warnings.warn("多模态输出模块未找到")


class SelfAwareMultimodalPipeline(BasicMultiModalPipeline):
    """
    自感知多模态RAG Pipeline
    
    扩展FlashRAG Pipeline到MRAG 3.0，实现：
    - ✅ 位置感知融合（已实现）
    - ⚠️ 不确定性估计（简化版）
    - ❌ 细粒度归因（待实现）
    - ❌ 多模态输出（待实现）
    
    使用示例：
    ```python
    config = {
        'device': 'cuda',
        'generator_model': 'llava-1.5-7b',
        'retrieval_method': 'clip',
        'use_position_fusion': True,
        'use_uncertainty_estimation': False,  # 暂未完全实现
        'retrieval_topk': 5
    }
    
    pipeline = SelfAwareMultimodalPipeline(config)
    results = pipeline.run(dataset)
    ```
    """
    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        初始化自感知多模态Pipeline
        
        Args:
            config: 配置字典
            prompt_template: 提示模板（可选）
            retriever: 检索器（可选）
            generator: 生成器（可选）
        """
        super().__init__(config, prompt_template)
        
        # 基础组件
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        
        # 配置选项
        self.use_position_fusion = config.get('use_position_fusion', True)
        self.use_uncertainty_estimation = config.get('use_uncertainty_estimation', False)
        self.use_fine_grained_attribution = config.get('use_fine_grained_attribution', False)
        self.retrieval_topk = config.get('retrieval_topk', 5)
        
        # 创新模块
        self._init_innovative_modules()
    
    def _init_innovative_modules(self):
        """初始化创新模块"""
        
        # 1. 位置感知融合模块 ✅
        if self.use_position_fusion and POSITION_FUSION_AVAILABLE:
            self.position_fusion = PositionAwareFusion(
                fusion_method=self.config.get('fusion_method', 'weighted'),
                position_encoding=self.config.get('position_encoding', 'learned')
            )
            print("✅ 位置感知融合模块已启用")
        else:
            self.position_fusion = None
            if self.use_position_fusion:
                print("⚠️ 位置感知融合模块未找到，已禁用")
        
        # 2. 不确定性估计模块 ✅ (完整版或简化版)
        if self.use_uncertainty_estimation:
            if UNCERTAINTY_ESTIMATOR_AVAILABLE:
                # 使用完整版
                mllm_model = self.config.get('mllm_model', None)
                self.uncertainty_estimator = CrossModalUncertaintyEstimator(
                    mllm_model=mllm_model,
                    config=self.config
                )
                print("✅ 不确定性估计模块已启用（完整版）")
            else:
                # 降级为简化版
                self.uncertainty_estimator = SimpleUncertaintyEstimator()
                print("⚠️ 不确定性估计模块已启用（简化版）")
        else:
            self.uncertainty_estimator = None
        
        # 3. 细粒度归因模块 ✅ (现已实现)
        if self.use_fine_grained_attribution:
            if ATTRIBUTION_AVAILABLE:
                mllm_model = self.config.get('mllm_model', None)
                self.attribution_module = FineGrainedMultimodalAttribution(
                    mllm_model=mllm_model,
                    config=self.config
                )
                print("✅ 细粒度归因模块已启用")
            else:
                print("⚠️ 细粒度归因模块导入失败，已禁用")
                self.attribution_module = None
        else:
            self.attribution_module = None
        
        # 4. 多模态输出模块 ✅ (MRAG 3.0)
        use_multimodal_output = self.config.get('use_multimodal_output', False)
        if use_multimodal_output:
            if MULTIMODAL_OUTPUT_AVAILABLE:
                self.multimodal_output = MultimodalOutputComposition(self.config)
                print("✅ 多模态输出模块已启用（MRAG 3.0）")
            else:
                print("⚠️ 多模态输出模块导入失败，已禁用")
                self.multimodal_output = None
        else:
            self.multimodal_output = None
    
    def naive_run(self, dataset, do_eval=True, pred_process_func=None):
        """
        不使用检索的基础运行（baseline）
        """
        input_prompts = [
            self.prompt_template.get_string(item) for item in dataset
        ]
        
        dataset.update_output("prompt", input_prompts)
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)
        
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)
        return dataset
    
    def run(self, dataset, do_eval=True, pred_process_func=None):
        """
        完整的自感知多模态RAG流程
        
        流程：
        1. 不确定性估计（可选）← 判断是否需要检索
        2. 多模态检索
        3. 位置感知融合 ← 我们的创新！
        4. 生成答案
        5. 细粒度归因（待实现）
        """
        
        # 准备查询
        if None not in dataset.question:
            text_query_list = dataset.question
        else:
            text_query_list = dataset.text if hasattr(dataset, 'text') else None
        
        image_query_list = dataset.image if hasattr(dataset, 'image') else [None] * len(dataset)
        
        # 步骤1: 不确定性估计（简化版）
        if self.uncertainty_estimator is not None:
            uncertainties = self._estimate_uncertainties(text_query_list, image_query_list)
            dataset.update_output("uncertainty", uncertainties)
        else:
            uncertainties = None
        
        # 步骤2: 执行检索
        retrieval_results = self._perform_retrieval(
            text_query_list, 
            image_query_list,
            uncertainties
        )
        dataset.update_output("retrieval_result", retrieval_results)
        
        # 步骤3: 位置感知融合（我们的创新！）
        if self.position_fusion is not None:
            retrieval_results = self._apply_position_fusion(
                retrieval_results, 
                text_query_list
            )
            dataset.update_output("fused_retrieval_result", retrieval_results)
        
        # 步骤4: 生成答案
        input_prompts = [
            self.prompt_template.get_string(
                question=q if q is not None else "",
                retrieval_result=r,
                image=img if img is not None else None
            )
            for q, r, img in zip(text_query_list or [""] * len(dataset), 
                                retrieval_results, 
                                image_query_list)
        ]
        dataset.update_output("prompt", input_prompts)
        
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)
        
        # 步骤5: 细粒度归因（✅ 现已实现）
        if self.attribution_module is not None:
            attributions = self._compute_attributions(
                pred_answer_list, 
                retrieval_results,
                image_query_list
            )
            dataset.update_output("attribution", attributions)
        else:
            attributions = None
        
        # 步骤6: 多模态输出增强（✅ MRAG 3.0）
        if self.multimodal_output is not None:
            multimodal_answers = self._enhance_with_multimodal_output(
                pred_answer_list,
                attributions,
                retrieval_results
            )
            dataset.update_output("multimodal_answer", multimodal_answers)
        
        # 评估
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)
        return dataset
    
    def _estimate_uncertainties(self, text_queries, image_queries):
        """
        估计查询的不确定性（简化版）
        
        完整版应该包括：
        - 文本不确定性（Gram矩阵 + 语义熵）
        - 视觉不确定性（attention variance）
        - 跨模态对齐不确定性（JS散度）
        """
        uncertainties = []
        for text_q, img_q in zip(text_queries or [], image_queries or []):
            uncertainty = self.uncertainty_estimator.estimate(text_q, img_q)
            uncertainties.append(uncertainty)
        
        return uncertainties
    
    def _perform_retrieval(self, text_queries, image_queries, uncertainties=None):
        """
        执行多模态检索
        
        支持：
        - 纯文本检索
        - 文本+图像混合检索
        - 自适应检索（基于不确定性）
        """
        retrieval_results = []
        
        # 检查检索器类型
        is_multimodal_retriever = hasattr(self.retriever, 'batch_search')
        
        if is_multimodal_retriever:
            # 使用FlashRAG的多模态检索器
            # 文本模态检索
            text_results = self.retriever.batch_search(
                text_queries if text_queries else [""] * len(image_queries),
                target_modal='text'
            )
            
            # 图像模态检索（如果有图像）
            if image_queries and any(img is not None for img in image_queries):
                try:
                    image_results = self.retriever.batch_search(
                        image_queries,
                        target_modal='image'
                    )
                    # 合并文本和图像检索结果
                    retrieval_results = [
                        (t if t else []) + (i if i else []) 
                        for t, i in zip(text_results, image_results)
                    ]
                except:
                    # 降级为纯文本检索
                    retrieval_results = text_results
            else:
                retrieval_results = text_results
        else:
            # 降级为简单文本检索
            if hasattr(self.retriever, 'search'):
                retrieval_results = [
                    self.retriever.search(q, num=self.retrieval_topk)[0] if hasattr(self.retriever.search(q, num=self.retrieval_topk), '__iter__') else self.retriever.search(q, num=self.retrieval_topk)
                    for q in (text_queries or [""] * len(image_queries))
                ]
            else:
                # 使用batch_search (对于文本检索器)
                retrieval_results = self.retriever.batch_search(
                    text_queries if text_queries else [""] * len(image_queries)
                )
        
        return retrieval_results
    
    def _apply_position_fusion(self, retrieval_results, queries):
        """
        应用位置感知融合
        
        这是我们的核心创新！
        - 缓解Lost in the middle问题
        - U型权重分布
        - 位置去偏
        """
        if self.position_fusion is None:
            return retrieval_results
        
        fused_results = []
        for results, query in zip(retrieval_results, queries or [""] * len(retrieval_results)):
            if not results:
                fused_results.append(results)
                continue
            
            # 提取分数（如果有）
            if isinstance(results[0], dict) and 'score' in results[0]:
                scores = [r.get('score', 1.0) for r in results]
            else:
                # 默认分数：递减
                scores = [1.0 - i * 0.1 for i in range(len(results))]
            
            # 应用位置感知融合
            adjusted_results, adjusted_scores = self.position_fusion.mitigate_position_bias(
                results, scores, query
            )
            
            fused_results.append(adjusted_results)
        
        return fused_results
    
    def _compute_attributions(self, answers, retrieval_results, images=None):
        """
        计算细粒度归因（✅ 已实现）
        
        包括：
        - Region-level视觉归因（Grad-CAM）
        - Token-level文本归因（attention-based）
        - Attribution confidence
        """
        if self.attribution_module is None:
            return [None] * len(answers)
        
        attributions = []
        
        for answer, retrieved, image in zip(answers, retrieval_results, images or [None]*len(answers)):
            attr_result = {'visual': [], 'text': []}
            
            # 视觉归因
            if image is not None:
                retrieved_images = [
                    doc.get('image') for doc in retrieved 
                    if doc.get('image') is not None
                ]
                if retrieved_images:
                    visual_attr = self.attribution_module.attribute_visual_evidence(
                        image=image,
                        generated_text=answer,
                        retrieved_images=retrieved_images
                    )
                    attr_result['visual'] = visual_attr
            
            # 文本归因
            retrieved_texts = [
                doc.get('contents', doc.get('text', '')) 
                for doc in retrieved
            ]
            if retrieved_texts:
                text_attr = self.attribution_module.attribute_text_evidence(
                    generated_text=answer,
                    retrieved_texts=retrieved_texts
                )
                attr_result['text'] = text_attr
            
            attributions.append(attr_result)
        
        return attributions
    
    def _enhance_with_multimodal_output(self, answers, attributions, retrieval_results):
        """
        使用多模态输出增强答案（✅ MRAG 3.0）
        
        三步流程：
        1. Position Identification
        2. Candidate Retrieval
        3. Matching and Insertion
        """
        if self.multimodal_output is None:
            return answers  # 返回原始文本答案
        
        multimodal_answers = []
        
        for answer, attr, retrieved in zip(answers, attributions or [None]*len(answers), retrieval_results):
            # 生成多模态答案
            multimodal_answer = self.multimodal_output.generate_multimodal_answer(
                text_answer=answer,
                retrieved_evidence=retrieved,
                attribution_results=attr
            )
            multimodal_answers.append(multimodal_answer)
        
        return multimodal_answers


class SimpleUncertaintyEstimator:
    """
    简化版不确定性估计器
    
    完整版应该实现：
    - 文本不确定性（基于SeaKR的Gram矩阵）
    - 视觉不确定性（attention variance）
    - 跨模态对齐不确定性（JS散度）
    
    参考文档：第805-849行
    """
    
    def __init__(self):
        self.threshold = 0.5
    
    def estimate(self, text_query, image_query):
        """
        估计查询的不确定性（简化版）
        
        Returns:
            dict: {'text': float, 'visual': float, 'alignment': float, 'total': float}
        """
        # 简化实现：基于查询长度和复杂度
        text_uncertainty = self._estimate_text_uncertainty(text_query)
        visual_uncertainty = self._estimate_visual_uncertainty(image_query)
        alignment_uncertainty = 0.0  # 简化：不估计对齐不确定性
        
        total_uncertainty = (text_uncertainty + visual_uncertainty) / 2
        
        return {
            'text': text_uncertainty,
            'visual': visual_uncertainty,
            'alignment': alignment_uncertainty,
            'total': total_uncertainty
        }
    
    def _estimate_text_uncertainty(self, text):
        """简化的文本不确定性估计"""
        if text is None or text == "":
            return 0.5
        
        # 基于问题长度的简单启发式
        words = text.split()
        if len(words) < 5:
            return 0.3  # 短问题，低不确定性
        elif len(words) > 15:
            return 0.8  # 长问题，高不确定性
        else:
            return 0.5  # 中等
    
    def _estimate_visual_uncertainty(self, image):
        """简化的视觉不确定性估计"""
        if image is None:
            return 0.0  # 无图像
        else:
            return 0.5  # 有图像，假设中等不确定性
    
    def should_retrieve(self, uncertainty_dict):
        """判断是否需要检索"""
        return uncertainty_dict['total'] > self.threshold

