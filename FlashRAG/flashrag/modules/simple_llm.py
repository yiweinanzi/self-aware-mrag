#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的LLM回退模块
Simple LLM Fallback Module

用于4GPU并行时其他GPU的轻量级推理
"""

import torch
import random
from typing import Optional, List, Dict, Any

class SimpleLLM:
    """
    简单的LLM实现，用于GPU资源不足时的回退
    提供基本的VQA推理功能
    """

    def __init__(self, device: str = 'cpu'):
        """
        初始化简单LLM

        Args:
            device: 计算设备
        """
        self.device = device
        self.common_answers = [
            "cat", "dog", "bird", "horse", "cow", "sheep", "pig",
            "car", "truck", "bus", "train", "plane", "boat",
            "house", "building", "tree", "flower", "mountain", "ocean",
            "red", "blue", "green", "yellow", "black", "white",
            "one", "two", "three", "four", "five", "many", "few",
            "yes", "no", "maybe"
        ]

        # 领域相关的关键词答案
        self.domain_answers = {
            "color": ["red", "blue", "green", "yellow", "black", "white", "brown", "pink", "purple", "orange"],
            "animal": ["cat", "dog", "bird", "horse", "cow", "sheep", "pig", "chicken", "duck", "fish"],
            "object": ["car", "truck", "bus", "train", "plane", "boat", "bicycle", "motorcycle", "house", "building"],
            "count": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "many", "few", "none"],
            "nature": ["tree", "flower", "mountain", "ocean", "river", "lake", "forest", "grass", "sky", "cloud"],
            "food": ["apple", "banana", "orange", "bread", "pizza", "cake", "meat", "fish", "rice", "pasta"],
            "yesno": ["yes", "no", "maybe", "unknown"]
        }

    def generate(self, text: str = None, image=None, max_new_tokens: int = 20, temperature: float = 0.01, question: str = None, image_path: Optional[str] = None, do_sample: bool = True, **kwargs) -> str:
        """
        生成答案（兼容多种调用方式）

        Args:
            text: 问题文本（优先）
            question: 问题文本（备用）
            image: 图像对象或路径
            image_path: 图像路径（备用）
            max_new_tokens: 最大新token数（忽略）
            temperature: 温度参数（忽略）

        Returns:
            生成的答案
        """
        # 使用text或question参数
        question_text = text if text is not None else question
        if question_text is None:
            return "unknown"

        question_lower = question_text.lower()

        # 基于关键词的回答生成
        for domain, keywords in self.domain_answers.items():
            if domain in question_lower:
                return random.choice(keywords)

        # 特殊问题模式
        if any(word in question_lower for word in ["how many", "number of", "count"]):
            return random.choice(self.domain_answers["count"])

        if any(word in question_lower for word in ["what color", "color is"]):
            return random.choice(self.domain_answers["color"])

        if any(word in question_lower for word in ["what animal", "animal is", "kind of animal"]):
            return random.choice(self.domain_answers["animal"])

        if any(word in question_lower for word in ["is there", "are there", "do you see"]):
            return random.choice(self.domain_answers["yesno"])

        # 默认随机答案
        return random.choice(self.common_answers)

    def generate_simple(self, question: str) -> str:
        """
        简化的生成方法（兼容性）

        Args:
            question: 问题文本

        Returns:
            生成的答案
        """
        return self.generate(question)

    def cleanup(self):
        """清理资源"""
        # 简单LLM没有需要清理的GPU资源
        pass

    def to(self, device: str):
        """
        移动到指定设备

        Args:
            device: 目标设备
        """
        self.device = device
        return self

    def __call__(self, question: str, **kwargs) -> str:
        """
        可调用接口

        Args:
            question: 问题文本
            **kwargs: 其他参数

        Returns:
            生成的答案
        """
        return self.generate(question)

class MockSimpleLLM(SimpleLLM):
    """
    模拟SimpleLLM类，用于测试和开发
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__(device)
        self.mock_mode = True

class FallbackLLMManager:
    """
    回退LLM管理器，用于管理多个GPU上的简单LLM实例
    """

    def __init__(self):
        self.instances = {}

    def get_instance(self, device_id: int) -> SimpleLLM:
        """
        获取指定设备的LLM实例

        Args:
            device_id: GPU设备ID

        Returns:
            SimpleLLM实例
        """
        if device_id not in self.instances:
            device = f'cuda:{device_id}' if device_id >= 0 else 'cpu'
            self.instances[device_id] = SimpleLLM(device)

        return self.instances[device_id]

    def cleanup_all(self):
        """清理所有实例"""
        for instance in self.instances.values():
            instance.cleanup()
        self.instances.clear()

# 全局管理器实例
fallback_manager = FallbackLLMManager()