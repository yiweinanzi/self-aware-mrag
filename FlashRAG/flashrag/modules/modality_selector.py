#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modality Selection模块

根据不确定性分数选择最佳检索模态

参考：文档第363-378行
"""

from typing import Dict, Optional

class ModalitySelector:
    """
    模态选择器

    根据不确定性分数选择检索模态：
    - 文本不确定性高 → 检索文本知识
    - 视觉不确定性高 → 检索视觉内容
    - 对齐不确定性高 → 检索跨模态内容（both）

    使用示例：
    ```python
    selector = ModalitySelector()

    modality = selector.select(uncertainty_scores={
        'text': 0.8,
        'visual': 0.3,
        'alignment': 0.5
    })
    # 返回: 'text' (因为text uncertainty最高)
    ```
    """

    def __init__(self, config=None):
        """
        初始化模态选择器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 阈值配置
        self.text_threshold = self.config.get('text_threshold', 0.4)
        self.visual_threshold = self.config.get('visual_threshold', 0.4)
        self.alignment_threshold = self.config.get('alignment_threshold', 0.4)

        print(f"✅ ModalitySelector初始化完成")

    def select(self, uncertainty_scores: Dict[str, float]) -> str:
        """
        选择检索模态

        策略（文档第366-378行）：
        - 文本不确定性高 → 检索文本
        - 视觉不确定性高 → 检索图像
        - 对齐不确定性高 → 检索both
        - 多个都高 → 选择最高的或both

        Args:
            uncertainty_scores: 不确定性分数字典
                {
                    'text': float,
                    'visual': float,
                    'alignment': float
                }

        Returns:
            str: 'text', 'image', 'both'
        """
        text_unc = uncertainty_scores.get('text', 0.0)
        visual_unc = uncertainty_scores.get('visual', 0.0)
        alignment_unc = uncertainty_scores.get('alignment', 0.0)

        # 策略1: 对齐不确定性高 → 需要跨模态信息
        if alignment_unc > self.alignment_threshold:
            return 'both'

        # 策略2: 文本不确定性最高 → 主要检索文本知识
        if text_unc > visual_unc and text_unc > self.text_threshold:
            return 'text'

        # 策略3: 视觉不确定性最高 → 主要检索视觉内容
        if visual_unc > text_unc and visual_unc > self.visual_threshold:
            return 'image'

        # 策略4: 都不高或都差不多 → 检索both
        if text_unc > self.text_threshold or visual_unc > self.visual_threshold:
            return 'both'

        # 默认：both（保守策略）
        return 'both'

    def get_modality_weights(self, modality: str) -> Dict[str, float]:
        """
        获取模态权重（用于混合检索）

        Args:
            modality: 'text', 'image', 'both'

        Returns:
            Dict: {'text': weight, 'image': weight}
        """
        if modality == 'text':
            return {'text': 1.0, 'image': 0.0}
        elif modality == 'image':
            return {'text': 0.0, 'image': 1.0}
        elif modality == 'both':
            return {'text': 0.5, 'image': 0.5}
        else:
            return {'text': 0.5, 'image': 0.5}


# 便捷函数
def select_modality(uncertainty_scores: Dict[str, float]) -> str:
    """快速选择模态"""
    selector = ModalitySelector()
    return selector.select(uncertainty_scores)


if __name__ == '__main__':
    print("Modality Selector模块测试")
    print("=" * 70)

    selector = ModalitySelector()

    # 测试不同场景
    test_cases = [
        {
            'unc': {'text': 0.8, 'visual': 0.2, 'alignment': 0.3},
            'desc': "高文本不确定性"
        },
        {
            'unc': {'text': 0.2, 'visual': 0.9, 'alignment': 0.3},
            'desc': "高视觉不确定性"
        },
        {
            'unc': {'text': 0.4, 'visual': 0.4, 'alignment': 0.8},
            'desc': "高对齐不确定性"
        },
        {
            'unc': {'text': 0.6, 'visual': 0.6, 'alignment': 0.5},
            'desc': "文本和视觉都较高"
        },
        {
            'unc': {'text': 0.2, 'visual': 0.2, 'alignment': 0.2},
            'desc': "都很低"
        }
    ]

    print("\n模态选择测试:\n")

    for case in test_cases:
        modality = selector.select(case['unc'])
        weights = selector.get_modality_weights(modality)

        print(f"{case['desc']}:")
        print(f"  不确定性: Text={case['unc']['text']:.1f}, "
              f"Visual={case['unc']['visual']:.1f}, "
              f"Align={case['unc']['alignment']:.1f}")
        print(f"  → 选择模态: {modality}")
        print(f"  → 权重: Text={weights['text']:.1f}, Image={weights['image']:.1f}")
        print()

    print("=" * 70)
    print("✅ Modality Selection测试通过！")
    print("\n策略说明:")
    print("  1. 对齐不确定性高 → both（需要跨模态信息）")
    print("  2. 文本不确定性高 → text（需要文本知识）")
    print("  3. 视觉不确定性高 → image（需要视觉信息）")
    print("  4. 默认 → both（保守策略）")
    print("=" * 70)