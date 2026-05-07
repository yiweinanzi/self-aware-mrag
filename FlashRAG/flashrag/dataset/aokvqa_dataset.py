#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A-OKVQA数据集加载器
A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge

论文: Schwenk et al., 2022
特点: 
- 24,903个多选题
- 需要外部知识和推理
- 包含rationales（推理过程）

参考文档：创新点1-自感知多模态RAG-实施方案.md 第1046-1050行
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

class AOKVQADataset:
    """
    A-OKVQA数据集加载器
    
    数据格式：
    {
        "question_id": str,
        "question": str,
        "choices": [str, str, str, str],  # 4个选项
        "correct_choice_idx": int,         # 正确选项索引（0-3）
        "image_id": str,
        "rationales": List[str] or None
    }
    
    使用示例：
    ```python
    dataset = AOKVQADataset({
        'data_dir': 'flashrag/data/A-OKVQA',
        'split': 'test'
    })
    
    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(sample.keys())
    ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化A-OKVQA数据集
        
        Args:
            config: 配置字典
                - data_dir: 数据目录
                - split: 'train', 'validation', 'test'
                - max_samples: 最大样本数（可选）
        """
        self.config = config
        self.data_dir = config.get('data_dir', 'flashrag/data/A-OKVQA')
        self.split = config.get('split', 'test')
        self.max_samples = config.get('max_samples', None)
        
        # 加载数据
        self.samples = self._load_data()
        
        print(f"✅ A-OKVQA数据集加载完成")
        print(f"   Split: {self.split}")
        print(f"   样本数: {len(self.samples)}")
    
    def _load_data(self) -> List[Dict]:
        """
        加载A-OKVQA数据
        
        Returns:
            List[Dict]: 样本列表
        """
        # 确定数据文件路径
        if self.split == 'train':
            sample_file = os.path.join(self.data_dir, 'train_sample.json')
            raw_file = os.path.join(self.data_dir, 'raw', 'train', 'data.json')
        elif self.split == 'validation' or self.split == 'val':
            sample_file = os.path.join(self.data_dir, 'validation_sample.json')
            raw_file = os.path.join(self.data_dir, 'raw', 'validation', 'data.json')
        else:  # test
            sample_file = os.path.join(self.data_dir, 'test_sample.json')
            raw_file = os.path.join(self.data_dir, 'raw', 'test', 'data.json')
        
        # 优先尝试加载raw文件（完整数据）
        if os.path.exists(raw_file):
            print(f"   从raw文件加载: {raw_file}")
            with open(raw_file) as f:
                data = json.load(f)
        elif os.path.exists(sample_file):
            print(f"   从sample文件加载: {sample_file}")
            with open(sample_file) as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"未找到A-OKVQA数据文件: {sample_file} 或 {raw_file}")
        
        # 转换为标准格式
        samples = []
        for item in data:
            sample = {
                'question_id': item['question_id'],
                'question': item['question'],
                'choices': item['choices'],
                'correct_choice_idx': item.get('correct_choice_idx'),
                'image_id': item.get('image_id', ''),
                'rationales': item.get('rationales', []),
                
                # 转换为统一格式（兼容其他数据集）
                'golden_answers': [item['choices'][item['correct_choice_idx']]] 
                                 if item.get('correct_choice_idx') is not None 
                                 else [],
                
                # 图像路径
                'image': self._get_image_path(item.get('image_id', ''))
            }
            samples.append(sample)
        
        # 限制样本数
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        return samples
    
    def _get_image_path(self, image_id: str) -> Optional[str]:
        """
        获取图像路径
        
        Args:
            image_id: 图像ID
            
        Returns:
            图像路径或None
        """
        if not image_id:
            return None
        
        # A-OKVQA使用COCO图像
        # 格式: COCO_val2014_000000XXXXXX.jpg
        # 尝试不同的图像目录
        possible_dirs = [
            os.path.join(self.data_dir, 'images'),
            os.path.join(self.data_dir, 'raw', 'images'),
            '/root/autodl-tmp/FlashRAG/flashrag/data/VQA/val2014',  # COCO图像可能在VQA目录
            '/root/autodl-tmp/FlashRAG/flashrag/data/VQA/train2014',
        ]
        
        for img_dir in possible_dirs:
            if os.path.exists(img_dir):
                # 尝试不同的文件名格式
                for ext in ['.jpg', '.png']:
                    # 格式1: 直接使用image_id
                    img_path = os.path.join(img_dir, f"{image_id}{ext}")
                    if os.path.exists(img_path):
                        return img_path
                    
                    # 格式2: COCO格式
                    img_path = os.path.join(img_dir, f"COCO_val2014_{image_id:012d}{ext}")
                    if os.path.exists(img_path):
                        return img_path
                    
                    # 格式3: 去掉前缀
                    if isinstance(image_id, str) and image_id.startswith('COCO_'):
                        img_path = os.path.join(img_dir, f"{image_id}{ext}")
                        if os.path.exists(img_path):
                            return img_path
        
        # 如果未找到，返回None（样本仍可用于纯文本测试）
        return None
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本
        
        Args:
            idx: 索引
            
        Returns:
            样本字典
        """
        if isinstance(idx, slice):
            return self.samples[idx]
        return self.samples[idx]
    
    def get_sample(self, idx: int) -> Dict:
        """获取单个样本（别名）"""
        return self.samples[idx]
    
    def evaluate_answer(self, predicted_answer: str, sample: Dict) -> bool:
        """
        评估答案是否正确
        
        对于多选题，检查预测是否包含正确选项
        
        Args:
            predicted_answer: 模型预测的答案
            sample: 样本字典
            
        Returns:
            bool: 是否正确
        """
        correct_idx = sample.get('correct_choice_idx')
        if correct_idx is None:
            return False
        
        correct_choice = sample['choices'][correct_idx]
        
        # 检查预测中是否包含正确选项
        pred_lower = predicted_answer.lower().strip()
        choice_lower = correct_choice.lower().strip()
        
        return choice_lower in pred_lower
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计字典
        """
        stats = {
            'total_samples': len(self.samples),
            'split': self.split,
            'has_rationales': sum(1 for s in self.samples if s.get('rationales')),
            'has_image': sum(1 for s in self.samples if s.get('image')),
            'has_correct_answer': sum(1 for s in self.samples if s.get('correct_choice_idx') is not None)
        }
        
        return stats


def create_aokvqa_dataset(config: Dict) -> AOKVQADataset:
    """
    工厂函数：创建A-OKVQA数据集
    
    Args:
        config: 配置字典
        
    Returns:
        AOKVQADataset实例
    """
    return AOKVQADataset(config)


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试A-OKVQA数据集加载器")
    print("=" * 80)
    
    # 测试配置
    config = {
        'data_dir': '/root/autodl-tmp/FlashRAG/flashrag/data/A-OKVQA',
        'split': 'test',
        'max_samples': 10
    }
    
    # 加载数据集
    dataset = AOKVQADataset(config)
    
    # 显示统计
    stats = dataset.get_statistics()
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 显示样本
    print(f"\n前3个样本:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Question: {sample['question']}")
        print(f"  Choices: {sample['choices']}")
        if sample.get('correct_choice_idx') is not None:
            print(f"  Correct: {sample['choices'][sample['correct_choice_idx']]}")
        print(f"  Image: {sample.get('image', 'None')}")
    
    print("\n" + "=" * 80)
    print("✅ A-OKVQA加载器测试完成")
    print("=" * 80)

