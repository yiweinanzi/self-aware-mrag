# -*- coding: utf-8 -*-
"""
OK-VQA数据集支持
扩展FlashRAG的数据集类

参考文档：创新点1-自感知多模态RAG-实施方案.md 第394-422行
"""

from flashrag.dataset.base_dataset import BaseDataset
from typing import List, Dict, Any, Optional
import json
from PIL import Image
import os
import warnings


class OKVQADataset(BaseDataset):
    """
    为FlashRAG添加OK-VQA支持
    
    遵循FlashRAG的数据格式，并添加多模态字段
    
    数据格式：
    {
        'id': question_id,
        'question': text question,
        'golden_answers': list of answers,
        'image': PIL.Image,  # 新增
        'image_id': image identifier  # 新增
    }
    
    使用示例：
    ```python
    from flashrag.dataset import OKVQADataset
    
    config = {
        'okvqa_data_path': 'data/okvqa/questions.json',
        'okvqa_image_path': 'data/okvqa/images/',
        'split': 'val'
    }
    
    dataset = OKVQADataset(config)
    ```
    """
    
    def __init__(self, config):
        """
        初始化OK-VQA数据集
        
        Args:
            config: 配置字典，需要包含：
                - okvqa_data_path: OK-VQA数据文件路径
                - okvqa_image_path: 图像文件夹路径
                - split: 'train'/'val'/'test'
        """
        super().__init__(config)
        self.name = 'okvqa'
        
        self.data_path = config.get('okvqa_data_path', 'data/okvqa/questions.json')
        self.image_path = config.get('okvqa_image_path', 'data/okvqa/images/')
        self.split = config.get('split', 'val')
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        加载OK-VQA原始数据并转换为FlashRAG格式
        
        Returns:
            List[Dict]: FlashRAG格式的数据
        """
        # 加载OK-VQA原始数据
        raw_data = self._load_okvqa_raw()
        
        # 转换为FlashRAG格式
        flashrag_format = []
        for sample in raw_data:
            flashrag_sample = {
                # FlashRAG标准字段
                'id': str(sample.get('question_id', sample.get('id', 'unknown'))),
                'question': sample.get('question', ''),
                'golden_answers': sample.get('answers', []),
                
                # 多模态扩展字段
                'image': self._load_image(sample.get('image_id', '')),
                'image_id': sample.get('image_id', ''),
                
                # 可选字段
                'metadata': {
                    'question_type': sample.get('question_type', 'unknown'),
                    'confidence': sample.get('confidence', [])
                }
            }
            flashrag_format.append(flashrag_sample)
        
        return flashrag_format
    
    def _load_okvqa_raw(self) -> List[Dict]:
        """
        加载OK-VQA原始JSON文件
        
        Returns:
            原始数据列表
        """
        if not os.path.exists(self.data_path):
            warnings.warn(f"数据文件不存在: {self.data_path}")
            return []
        
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            # OK-VQA格式转换
            # 原始格式：{'questions': [...], 'annotations': [...]}
            # 需要根据实际OK-VQA格式调整
            
            if isinstance(data, dict):
                if 'questions' in data:
                    questions = data['questions']
                elif 'annotations' in data:
                    questions = data['annotations']
                else:
                    # 假设直接是列表
                    questions = [data]
            elif isinstance(data, list):
                questions = data
            else:
                warnings.warn(f"未知的数据格式: {type(data)}")
                return []
            
            return questions
        
        except Exception as e:
            warnings.warn(f"加载OK-VQA数据失败: {e}")
            return []
    
    def _load_image(self, image_id: str) -> Optional[Image.Image]:
        """
        加载图像
        
        Args:
            image_id: 图像ID
            
        Returns:
            PIL.Image对象或None
        """
        if not image_id:
            return None
        
        # OK-VQA图像命名格式：COCO_val2014_000000xxxxxx.jpg
        try:
            image_id_int = int(image_id)
            image_file = f"COCO_{self.split}2014_{image_id_int:012d}.jpg"
        except ValueError:
            # 如果不是整数，直接使用
            image_file = image_id if image_id.endswith('.jpg') else f"{image_id}.jpg"
        
        image_path = os.path.join(self.image_path, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            warnings.warn(f"无法加载图像 {image_path}: {e}")
            # 返回空白图像作为占位符
            return Image.new('RGB', (224, 224), color='white')


class WebQADataset(BaseDataset):
    """
    WebQA数据集支持
    
    参考文档第1014-1021行
    
    WebQA特点：
    - 24,929 image-based questions
    - 24,343 text-based questions
    - Multihop and multimodal QA
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'webqa'
        
        self.data_path = config.get('webqa_data_path', 'data/webqa/questions.json')
        self.image_path = config.get('webqa_image_path', 'data/webqa/images/')
        self.split = config.get('split', 'val')
    
    def load_data(self) -> List[Dict]:
        """
        加载WebQA数据
        
        Returns:
            List[Dict]: FlashRAG格式的数据
        """
        # TODO: 根据实际WebQA格式实现
        if not os.path.exists(self.data_path):
            warnings.warn(f"WebQA数据文件不存在: {self.data_path}")
            return []
        
        try:
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
            
            # 转换为FlashRAG格式
            flashrag_format = []
            for sample in raw_data:
                flashrag_sample = {
                    'id': str(sample.get('guid', sample.get('id', 'unknown'))),
                    'question': sample.get('question', ''),
                    'golden_answers': sample.get('answer', []),
                    'image': self._load_image(sample.get('img_id', '')),
                    'image_id': sample.get('img_id', ''),
                    'metadata': sample.get('metadata', {})
                }
                flashrag_format.append(flashrag_sample)
            
            return flashrag_format
        
        except Exception as e:
            warnings.warn(f"加载WebQA数据失败: {e}")
            return []
    
    def _load_image(self, image_id: str) -> Optional[Image.Image]:
        """加载图像"""
        if not image_id:
            return None
        
        image_path = os.path.join(self.image_path, f"{image_id}.jpg")
        
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            warnings.warn(f"无法加载图像: {e}")
            return Image.new('RGB', (224, 224), color='white')


class MultiModalQADataset(BaseDataset):
    """
    MultiModalQA数据集支持
    
    参考文档第1023-1029行
    
    MultiModalQA特点：
    - 29,918 questions
    - 35.7% require cross-modality reasoning
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'multimodalqa'
        
        self.data_path = config.get('multimodalqa_data_path', 'data/multimodalqa/data.json')
        self.image_path = config.get('multimodalqa_image_path', 'data/multimodalqa/images/')
        self.split = config.get('split', 'dev')
    
    def load_data(self) -> List[Dict]:
        """
        加载MultiModalQA数据
        
        Returns:
            List[Dict]: FlashRAG格式的数据
        """
        # TODO: 根据实际MultiModalQA格式实现
        if not os.path.exists(self.data_path):
            warnings.warn(f"MultiModalQA数据文件不存在: {self.data_path}")
            return []
        
        try:
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
            
            flashrag_format = []
            for sample in raw_data:
                flashrag_sample = {
                    'id': str(sample.get('qid', 'unknown')),
                    'question': sample.get('question', ''),
                    'golden_answers': sample.get('answers', []),
                    'image': self._load_image(sample.get('image_id', '')),
                    'image_id': sample.get('image_id', ''),
                    'metadata': {
                        'modality': sample.get('modality', 'unknown'),
                        'hops': sample.get('num_hops', 1)
                    }
                }
                flashrag_format.append(flashrag_sample)
            
            return flashrag_format
        
        except Exception as e:
            warnings.warn(f"加载MultiModalQA数据失败: {e}")
            return []
    
    def _load_image(self, image_id: str) -> Optional[Image.Image]:
        """加载图像"""
        if not image_id:
            return None
        
        image_path = os.path.join(self.image_path, f"{image_id}.jpg")
        
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            return Image.new('RGB', (224, 224), color='white')


# 工厂函数
def create_multimodal_dataset(dataset_name: str, config: Dict) -> BaseDataset:
    """
    创建多模态数据集
    
    Args:
        dataset_name: 'okvqa', 'webqa', 'multimodalqa'
        config: 配置字典
        
    Returns:
        对应的数据集实例
    """
    if dataset_name.lower() == 'okvqa':
        return OKVQADataset(config)
    elif dataset_name.lower() == 'webqa':
        return WebQADataset(config)
    elif dataset_name.lower() == 'multimodalqa':
        return MultiModalQADataset(config)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")


def load_okvqa_dataset(split: str = 'val', max_samples: Optional[int] = None):
    """
    便捷的OK-VQA数据集加载函数

    Args:
        split: 数据集分割 ('train', 'val', 'test')
        max_samples: 最大样本数

    Returns:
        List[Dict]: 数据列表
    """
    config = {
        'okvqa_data_path': '/data1/userdata/zqwang/ACL_data/OK-VQA',
        'okvqa_image_path': '/data1/userdata/zqwang/ACL_data/OK-VQA/images',
        'split': split,
        'max_samples': max_samples
    }

    # 创建示例数据（如果真实数据不存在）
    if not os.path.exists(config['okvqa_data_path']):
        warnings.warn(f"OK-VQA数据路径不存在: {config['okvqa_data_path']}，使用示例数据")
        sample_data = []

        for i in range(min(max_samples or 10, 10)):
            sample_data.append({
                'question': f'Sample OK-VQA question {i+1}?',
                'answer': f'answer {i+1}',
                'golden_answers': [f'answer {i+1}'],
                'image': None,
                'image_id': f'sample_{i+1}'
            })

        return sample_data

    # 使用真实数据集类加载
    dataset = OKVQADataset(config)
    data = dataset.load_data()

    if max_samples:
        data = data[:max_samples]

    return data
