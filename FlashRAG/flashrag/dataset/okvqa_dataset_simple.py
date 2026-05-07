# -*- coding: utf-8 -*-
"""
OK-VQA数据集加载器（简化版，不依赖torch）
用于内存受限的环境
"""

import os
import json
from typing import List, Dict, Any
from PIL import Image


class OKVQADatasetSimple:
    """
    简化版OK-VQA数据集加载器
    不依赖torch.utils.data.Dataset，降低内存占用
    """
    
    def __init__(self, config):
        """
        Args:
            config: 配置字典
                - data_dir: 数据集根目录
                - split: 'train'/'val'
                - load_images: 是否加载图像
        """
        self.data_dir = config.get('data_dir', 'flashrag/data/VQA')
        self.split = config.get('split', 'val')
        default_image_dir = os.path.join(self.data_dir, f'{self.split}2014')
        self.image_dir = config.get('image_dir', default_image_dir)
        self.load_images = config.get('load_images', True)
        self.name = 'okvqa'
        
        # 加载数据
        self.data = self.load_data()
    
    def load_data(self):
        """加载数据并转换为标准格式"""
        # 加载问题和标注
        raw_data = self.load_okvqa_raw()
        
        if not raw_data:
            print(f"⚠️ 未加载到任何数据")
            return []
        
        # 转换为标准格式
        formatted_data = []
        images_loaded = 0
        
        for idx, sample in enumerate(raw_data):
            # 加载图像（如果需要）
            image = None
            if self.load_images:
                image = self._load_image(sample['image_id'])
                if image is not None:
                    images_loaded += 1
            
            formatted_data.append({
                'id': str(sample['question_id']),
                'question': sample['question'],
                'golden_answers': sample['answers'],
                'image': image,
                'image_id': sample['image_id'],
                'metadata': {
                    'dataset': 'okvqa',
                    'split': self.split
                }
            })
        
        print(f"✅ 成功加载 {len(formatted_data)} 个样本")
        if self.load_images:
            print(f"   其中 {images_loaded} 个样本加载了图像 ({images_loaded/len(formatted_data)*100:.1f}%)")
        
        return formatted_data
    
    def load_okvqa_raw(self):
        """加载OK-VQA原始数据文件"""
        question_file = os.path.join(
            self.data_dir,
            f'OpenEnded_mscoco_{self.split}2014_questions.json'
        )
        annotation_file = os.path.join(
            self.data_dir,
            f'mscoco_{self.split}2014_annotations.json'
        )
        
        # 加载问题
        if not os.path.exists(question_file):
            print(f"❌ 问题文件不存在: {question_file}")
            return []
        
        with open(question_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            questions = questions_data.get('questions', questions_data)
        
        # 加载标注
        annotations = {}
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                anno_data = json.load(f)
                anno_list = anno_data.get('annotations', anno_data)
                annotations = {item['question_id']: item for item in anno_list}
        
        # 合并问题和答案
        raw_data = []
        for q in questions:
            question_id = q['question_id']
            anno = annotations.get(question_id, {})
            
            # 提取答案列表
            answers = []
            if 'answers' in anno:
                answers = [a['answer'] for a in anno['answers']]
            elif 'answer' in anno:
                answers = [anno['answer']] if isinstance(anno['answer'], str) else anno['answer']
            
            raw_data.append({
                'question_id': question_id,
                'question': q['question'],
                'image_id': q['image_id'],
                'answers': answers if answers else ['unknown']
            })
        
        return raw_data
    
    def _load_image(self, image_id):
        """加载图像文件"""
        if not self.load_images or not os.path.exists(self.image_dir):
            return None
        
        # COCO格式的图像文件名
        image_filename = f'COCO_{self.split}2014_{str(image_id).zfill(12)}.jpg'
        image_path = os.path.join(self.image_dir, image_filename)
        
        if os.path.exists(image_path):
            try:
                return Image.open(image_path).convert('RGB')
            except Exception as e:
                return None
        
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __repr__(self):
        return f"<OKVQADatasetSimple name={self.name} size={len(self)}>"

