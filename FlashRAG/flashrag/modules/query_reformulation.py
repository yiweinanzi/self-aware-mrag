#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query Reformulation模块

根据文档第88-111行的要求实现查询重构

功能：
- 根据不确定性类型调整查询
- 基于模态选择重构查询
- 增强检索效果

参考：创新点1-自感知多模态RAG-实施方案.md 第109-111行
"""

from typing import Dict, Optional

class QueryReformulator:
    """
    查询重构器
    
    根据不确定性分数重构查询，提高检索质量
    
    使用示例：
    ```python
    reformulator = QueryReformulator()
    
    # 基于不确定性重构查询
    enhanced_query = reformulator.reformulate(
        query="What is this?",
        uncertainty_scores={'text': 0.8, 'visual': 0.3, 'alignment': 0.6}
    )
    ```
    """
    
    def __init__(self, config=None):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        print("✅ QueryReformulator初始化完成")
    
    def reformulate(self, 
                   query: str,
                   uncertainty_scores: Dict[str, float],
                   modality: Optional[str] = None) -> str:
        """
        重构查询 - 完整版实现
        
        根据不确定性类型智能调整查询，保留关键信息，使其更适合检索
        
        创新点：
        1. 保留多选题选项等关键结构信息
        2. 根据不确定性类型提取核心实体和关键词
        3. 避免过长查询导致的检索效果下降
        
        Args:
            query: 原始查询
            uncertainty_scores: 不确定性分数字典
            modality: 检索模态 ('text', 'image', 'both')
            
        Returns:
            str: 重构后的查询
        """
        # ========== 第1步: 分析查询结构 ==========
        is_multiple_choice = ('Options:' in query or 
                             ('A.' in query and 'B.' in query and 'C.' in query))
        
        if is_multiple_choice:
            # 多选题：提取核心问题，保留选项
            lines = query.strip().split('\n')
            core_question = lines[0] if lines else query
            
            # 提取选项关键信息
            option_keywords = []
            for line in lines:
                if line.strip().startswith(('A.', 'B.', 'C.', 'D.')):
                    # 提取选项内容（去除标签）
                    option_text = line.split('.', 1)[1].strip() if '.' in line else line
                    # 只保留前30个字符作为关键词
                    option_keywords.append(option_text[:30])
            
            # 构建增强查询：核心问题 + 选项关键词
            if option_keywords:
                core_question_clean = core_question.replace('Answer with the letter only (A/B/C/D):', '').strip()
                enhanced_query = f"{core_question_clean} Options: {' / '.join(option_keywords)}"
            else:
                enhanced_query = core_question
        else:
            # 普通问题：保持原样
            enhanced_query = query
        
        # ========== 第2步: 根据不确定性添加检索提示 ==========
        text_unc = uncertainty_scores.get('text', 0)
        visual_unc = uncertainty_scores.get('visual', 0)
        align_unc = uncertainty_scores.get('alignment', 0)
        
        # 判断主要不确定性来源
        if text_unc > 0.5:
            # 文本不确定性高：需要事实性知识
            search_hint = " [需要事实知识]"
        elif visual_unc > 0.5:
            # 视觉不确定性高：需要视觉描述
            search_hint = " [需要视觉描述]"
        elif align_unc > 0.5:
            # 对齐不确定性高：需要多模态信息
            search_hint = " [需要图文信息]"
        else:
            search_hint = ""
        
        # ========== 第3步: 长度控制 ==========
        # 避免查询过长影响检索效果（限制在200字符）
        if len(enhanced_query) > 200:
            enhanced_query = enhanced_query[:200].rsplit(' ', 1)[0] + "..."
        
        return enhanced_query + search_hint
    
    def reformulate_for_modality(self,
                                 query: str,
                                 modality: str) -> str:
        """
        根据检索模态重构查询 - 完整版实现
        
        Args:
            query: 原始查询
            modality: 'text', 'image', 'both'
            
        Returns:
            str: 适配特定模态的查询
        """
        if modality == 'text':
            # 文本检索：提取核心事实性问题
            # 移除视觉相关的提示词
            query_clean = query.replace('[需要视觉描述]', '').replace('[需要图文信息]', '').strip()
            return query_clean
        
        elif modality == 'image':
            # 图像检索：强调视觉特征和描述
            # 移除纯文本知识提示
            query_clean = query.replace('[需要事实知识]', '').strip()
            return query_clean
        
        elif modality == 'both':
            # 跨模态检索：保持原查询的完整性
            return query
        
        else:
            return query


# 便捷函数
def reformulate_query(query: str, uncertainty_scores: Dict[str, float]) -> str:
    """快速重构查询"""
    reformulator = QueryReformulator()
    return reformulator.reformulate(query, uncertainty_scores)


if __name__ == '__main__':
    print("Query Reformulation模块")
    print("=" * 70)
    print("功能：根据不确定性重构查询\n")
    
    reformulator = QueryReformulator()
    
    # 测试不同不确定性场景
    test_cases = [
        {
            'query': "Who invented the telephone?",
            'unc': {'text': 0.8, 'visual': 0.2, 'alignment': 0.3},
            'desc': "高文本不确定性"
        },
        {
            'query': "What color is this car?",
            'unc': {'text': 0.2, 'visual': 0.9, 'alignment': 0.4},
            'desc': "高视觉不确定性"
        },
        {
            'query': "What is shown in the image?",
            'unc': {'text': 0.4, 'visual': 0.4, 'alignment': 0.8},
            'desc': "高对齐不确定性"
        }
    ]
    
    for case in test_cases:
        enhanced = reformulator.reformulate(case['query'], case['unc'])
        print(f"\n{case['desc']}:")
        print(f"  原始: {case['query']}")
        print(f"  增强: {enhanced}")
    
    print("\n" + "=" * 70)
    print("✅ Query Reformulation实现完成！")


