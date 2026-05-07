# -*- coding: utf-8 -*-
"""
多模态输出组合模块
Multimodal Output Composition

符合MRAG 3.0的End-to-End Multimodality要求

实现三步流程（华为综述Section 2.3.2）：
1. Position Identification - 判断哪里插入图像
2. Candidate Set Retrieval - 检索候选图像
3. Matching and Insertion - 匹配并插入

参考文档：创新点1-自感知多模态RAG-实施方案.md 第42-145行
"""

import warnings
from typing import List, Dict, Any, Optional, Tuple

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL未安装，多模态输出功能不可用")


class MultimodalOutputComposition:
    """
    多模态输出组合模块
    
    符合MRAG 3.0的多模态输出要求
    参考：华为综述Fig.5的三种场景
    
    Sub-scenarios:
    - Sub-scenario I: 纯图像回答
    - Sub-scenario II: 图文混合step-by-step
    - Sub-scenario III: 丰富性增强
    
    使用示例：
    ```python
    composer = MultimodalOutputComposition()
    
    multimodal_answer = composer.generate_multimodal_answer(
        text_answer="The Eiffel Tower is in Paris...",
        retrieved_evidence=retrieved_docs,
        attribution_results=attributions
    )
    
    # multimodal_answer包含:
    # {
    #     'text': "The Eiffel Tower is in Paris...",
    #     'images': [img1, img2],
    #     'insertion_points': [10, 25],
    #     'scenario': 'Sub-scenario III'
    # }
    ```
    """
    
    def __init__(self, config=None):
        """
        初始化多模态输出模块
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 配置参数
        self.confidence_threshold = self.config.get('insertion_confidence_threshold', 0.7)
        self.max_images_per_answer = self.config.get('max_images_per_answer', 3)
        
        # 场景检测阈值
        self.pure_image_threshold = self.config.get('pure_image_threshold', 0.9)
        self.mixed_threshold = self.config.get('mixed_threshold', 0.5)
    
    def generate_multimodal_answer(self, text_answer: str,
                                   retrieved_evidence: List[Dict],
                                   attribution_results: Optional[List[Dict]] = None) -> Dict:
        """
        生成多模态答案
        
        三步流程：
        1. Position Identification
        2. Candidate Set Retrieval
        3. Matching and Insertion
        
        Args:
            text_answer: 文本答案
            retrieved_evidence: 检索到的证据
            attribution_results: 归因结果（可选）
            
        Returns:
            dict: 多模态答案
                {
                    'text': str,
                    'images': List[Image],
                    'insertion_points': List[int],
                    'scenario': str
                }
        """
        # 检测场景类型
        scenario = self._detect_scenario(text_answer, retrieved_evidence)
        
        if scenario == 'Sub-scenario I':
            # 纯图像回答
            return self._generate_pure_image_answer(retrieved_evidence)
        
        elif scenario == 'Sub-scenario II':
            # 图文混合step-by-step
            return self._generate_mixed_stepwise_answer(
                text_answer, retrieved_evidence, attribution_results
            )
        
        else:  # Sub-scenario III
            # 丰富性增强
            return self._generate_enriched_answer(
                text_answer, retrieved_evidence, attribution_results
            )
    
    def enhance(self, text_answer: str, attributions: Optional[List[Dict]],
               retrieved_evidence: List[Dict]) -> Dict:
        """
        增强文本答案为多模态答案
        
        这是主要接口，用于Pipeline集成
        
        Args:
            text_answer: 文本答案
            attributions: 归因结果
            retrieved_evidence: 检索证据
            
        Returns:
            多模态答案字典
        """
        return self.generate_multimodal_answer(
            text_answer, retrieved_evidence, attributions
        )
    
    # =========================================================================
    # Step 1: Position Identification
    # =========================================================================
    
    def identify_insertion_points(self, text_answer: str,
                                  attribution_results: Optional[Dict] = None) -> List[int]:
        """
        判断在哪里插入图像
        
        ✅ 修正：正确处理归因格式 {'visual': [...], 'text': [...]}
        
        策略：
        1. 利用归因结果（如果有）✅ 修正格式匹配
        2. 检测描述性句子
        3. 识别关键名词
        
        Args:
            text_answer: 文本答案
            attribution_results: 归因结果（格式：{'visual': [...], 'text': [...]}）
            
        Returns:
            List[int]: 插入位置（字符索引）
        """
        insertion_points = []
        
        # 策略1：基于归因结果 ✅ 修正
        if attribution_results and isinstance(attribution_results, dict):
            # ✅ 正确处理归因格式：{'visual': [...], 'text': [...]}
            
            # 1a. 使用视觉归因
            if 'visual' in attribution_results and attribution_results['visual']:
                for visual_attr in attribution_results['visual']:
                    confidence = visual_attr.get('confidence', 0)
                    
                    if confidence > self.confidence_threshold:
                        # 方案1：如果有bbox信息，推断对应的文本位置
                        bbox = visual_attr.get('region_bbox')
                        if bbox:
                            text_pos = self._bbox_to_text_position(bbox, text_answer)
                            if text_pos is not None:
                                insertion_points.append(text_pos)
                        else:
                            # 方案2：使用source_image_id匹配
                            # 简化：在句子结尾插入
                            sentences = text_answer.split('.')
                            if sentences:
                                pos = len(sentences[0]) + 1
                                insertion_points.append(pos)
            
            # 1b. 使用文本归因推断视觉补充位置
            if 'text' in attribution_results and attribution_results['text']:
                # 在有高置信度文本归因的地方，可能需要视觉补充
                for text_attr in attribution_results['text']:
                    position = text_attr.get('position')  # token位置
                    confidence = text_attr.get('confidence', 0)
                    
                    if position is not None and confidence > 0.8:
                        # 将token位置转换为字符位置
                        char_pos = self._token_position_to_char_position(
                            position, text_answer
                        )
                        if char_pos is not None:
                            insertion_points.append(char_pos)
        
        # 策略2：检测句子边界（如果没有归因结果或归因为空）
        if not insertion_points:
            sentences = text_answer.split('.')
            cumulative_length = 0
            for idx, sent in enumerate(sentences):
                cumulative_length += len(sent) + 1  # +1 for the period
                # 在前几个句子后插入（不要全部插入）
                if cumulative_length < len(text_answer) and idx < 2:
                    insertion_points.append(cumulative_length)
        
        # 去重并排序
        insertion_points = sorted(list(set(insertion_points)))
        
        # 限制数量
        insertion_points = insertion_points[:self.max_images_per_answer]
        
        return insertion_points
    
    def _bbox_to_text_position(self, bbox: List[int], text: str) -> Optional[int]:
        """
        将region bbox转换为文本中的插入位置
        
        完整版应该：
        1. 使用语义分析找到描述该bbox的句子
        2. 返回该句子的结束位置
        
        简化版：返回第一个句子的结束位置
        
        Args:
            bbox: [x, y, w, h]
            text: 文本答案
            
        Returns:
            插入位置（字符索引）或None
        """
        # TODO: 实现语义匹配
        # 这需要：
        # 1. 提取bbox对应的视觉内容描述
        # 2. 在text中找到匹配的句子
        # 3. 返回该句子的位置
        
        # 简化版：返回第一个句子末尾
        sentences = text.split('.')
        if sentences:
            return len(sentences[0]) + 1
        
        return None
    
    def _token_position_to_char_position(self, token_pos: int, text: str) -> Optional[int]:
        """
        将token位置转换为字符位置
        
        Args:
            token_pos: token索引
            text: 文本
            
        Returns:
            字符索引
        """
        # 简单分词
        tokens = text.split()
        
        if token_pos >= len(tokens):
            return None
        
        # 计算前token_pos个token的累积长度
        char_pos = sum(len(t) + 1 for t in tokens[:token_pos])  # +1 for space
        
        return char_pos
    
    # =========================================================================
    # Step 2: Candidate Set Retrieval
    # =========================================================================
    
    def retrieve_visual_supplements(self, text_answer: str,
                                    insertion_points: List[int],
                                    retrieved_evidence: List[Dict]) -> List[Dict]:
        """
        检索候选视觉元素
        
        使用已检索的图像作为候选集
        
        Args:
            text_answer: 文本答案
            insertion_points: 插入位置
            retrieved_evidence: 已检索的证据
            
        Returns:
            List[Dict]: 候选图像列表
        """
        candidates = []
        
        # 从检索证据中提取图像
        for doc in retrieved_evidence:
            if 'image' in doc and doc['image'] is not None:
                candidates.append({
                    'image': doc['image'],
                    'source_id': doc.get('id', 'unknown'),
                    'relevance_score': doc.get('score', 1.0)
                })
        
        # 按相关度排序
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # 限制数量
        candidates = candidates[:len(insertion_points)]
        
        return candidates
    
    # =========================================================================
    # Step 3: Matching and Insertion
    # =========================================================================
    
    def insert_visual_elements(self, text_answer: str,
                               insertion_points: List[int],
                               candidates: List[Dict]) -> Dict:
        """
        匹配并插入视觉元素
        
        Args:
            text_answer: 文本答案
            insertion_points: 插入位置
            candidates: 候选图像
            
        Returns:
            dict: 多模态答案
        """
        # 匹配候选图像到插入点
        matched_images = []
        for idx, point in enumerate(insertion_points):
            if idx < len(candidates):
                matched_images.append(candidates[idx]['image'])
        
        return {
            'text': text_answer,
            'images': matched_images,
            'insertion_points': insertion_points,
            'num_images': len(matched_images)
        }
    
    # =========================================================================
    # 场景检测
    # =========================================================================
    
    def _detect_scenario(self, text_answer: str, 
                        retrieved_evidence: List[Dict]) -> str:
        """
        检测适用的Sub-scenario
        
        Sub-scenario I: 纯图像回答（如"识别这个物体"）
        Sub-scenario II: 图文混合step-by-step（如操作指南）
        Sub-scenario III: 丰富性增强（如介绍景点）
        """
        # 简化判断：基于文本长度和关键词
        text_length = len(text_answer.split())
        
        # 检测是否是识别类问题
        identification_keywords = ['is', 'are', 'object', 'thing', 'item']
        if text_length < 5 and any(kw in text_answer.lower() for kw in identification_keywords):
            return 'Sub-scenario I'
        
        # 检测是否包含步骤
        stepwise_keywords = ['first', 'second', 'then', 'next', 'finally', 'step']
        if any(kw in text_answer.lower() for kw in stepwise_keywords):
            return 'Sub-scenario II'
        
        # 默认为丰富性增强
        return 'Sub-scenario III'
    
    def _generate_pure_image_answer(self, retrieved_evidence: List[Dict]) -> Dict:
        """
        生成纯图像回答（Sub-scenario I）
        
        适用场景：识别类问题
        """
        # 提取最相关的图像
        images = []
        for doc in retrieved_evidence[:1]:  # 只取最相关的
            if 'image' in doc and doc['image'] is not None:
                images.append(doc['image'])
        
        return {
            'text': "",  # 无文本
            'images': images,
            'insertion_points': [],
            'scenario': 'Sub-scenario I'
        }
    
    def _generate_mixed_stepwise_answer(self, text_answer: str,
                                        retrieved_evidence: List[Dict],
                                        attribution_results: Optional[List[Dict]]) -> Dict:
        """
        生成图文混合step-by-step答案（Sub-scenario II）
        
        适用场景：操作指南、教程等
        """
        # 检测步骤位置
        sentences = text_answer.split('.')
        insertion_points = []
        
        for idx, sent in enumerate(sentences):
            # 在每个步骤后插入图像
            if any(kw in sent.lower() for kw in ['first', 'second', 'then', 'step']):
                # 计算字符位置
                pos = sum(len(s) + 1 for s in sentences[:idx+1])
                insertion_points.append(pos)
        
        # 检索候选图像
        candidates = self.retrieve_visual_supplements(
            text_answer, insertion_points, retrieved_evidence
        )
        
        # 插入
        return self.insert_visual_elements(
            text_answer, insertion_points, candidates
        )
    
    def _generate_enriched_answer(self, text_answer: str,
                                  retrieved_evidence: List[Dict],
                                  attribution_results: Optional[List[Dict]]) -> Dict:
        """
        生成丰富性增强答案（Sub-scenario III）
        
        适用场景：介绍性内容，需要视觉补充
        """
        # 识别插入点
        insertion_points = self.identify_insertion_points(
            text_answer, attribution_results
        )
        
        # 检索候选
        candidates = self.retrieve_visual_supplements(
            text_answer, insertion_points, retrieved_evidence
        )
        
        # 插入
        return self.insert_visual_elements(
            text_answer, insertion_points, candidates
        )


# 工厂函数
def create_multimodal_output_module(config=None):
    """
    创建多模态输出模块
    
    Args:
        config: 配置字典
        
    Returns:
        MultimodalOutputComposition实例
    """
    return MultimodalOutputComposition(config)

