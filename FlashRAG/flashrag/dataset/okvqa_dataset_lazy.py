# -*- coding: utf-8 -*-
"""
OK-VQA数据集加载器（懒加载版本，极低内存占用）
只在需要时才加载具体样本
"""

import os
import json
from PIL import Image


class OKVQADatasetLazy:
    """
    懒加载版OK-VQA数据集
    只在内存中保持索引，按需加载样本
    """
    
    def __init__(self, config):
        self.data_dir = config.get('data_dir', 'flashrag/data/VQA')
        self.split = config.get('split', 'val')
        default_image_dir = os.path.join(self.data_dir, f'{self.split}2014')
        self.image_dir = config.get('image_dir', default_image_dir)
        self.load_images = config.get('load_images', False)
        self.name = 'okvqa'
        
        # 只加载索引，不加载全部数据
        self._load_index()
    
    def _load_index(self):
        """只加载问题和答案的ID映射，不加载全部内容"""
        question_file = os.path.join(
            self.data_dir,
            f'OpenEnded_mscoco_{self.split}2014_questions.json'
        )
        annotation_file = os.path.join(
            self.data_dir,
            f'mscoco_{self.split}2014_annotations.json'
        )
        
        print(f"📂 加载索引: {question_file}")
        
        # 加载问题
        with open(question_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            self.questions = questions_data.get('questions', questions_data)
        
        # 加载标注
        with open(annotation_file, 'r', encoding='utf-8') as f:
            anno_data = json.load(f)
            anno_list = anno_data.get('annotations', anno_data)
            self.annotations = {item['question_id']: item for item in anno_list}
        
        print(f"✅ 索引加载完成: {len(self.questions)} 个样本")
    
    def _load_image(self, image_id):
        """懒加载图像"""
        if not self.load_images or not os.path.exists(self.image_dir):
            return None
        
        image_filename = f'COCO_{self.split}2014_{str(image_id).zfill(12)}.jpg'
        image_path = os.path.join(self.image_dir, image_filename)
        
        if os.path.exists(image_path):
            try:
                return Image.open(image_path).convert('RGB')
            except:
                return None
        return None
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        """按需加载单个样本"""
        q = self.questions[idx]
        question_id = q['question_id']
        anno = self.annotations.get(question_id, {})
        
        # 提取答案
        answers = []
        if 'answers' in anno:
            answers = [a['answer'] for a in anno['answers']]
        
        # 懒加载图像
        image = self._load_image(q['image_id']) if self.load_images else None
        
        return {
            'id': str(question_id),
            'question': q['question'],
            'golden_answers': answers if answers else ['unknown'],
            'image': image,
            'image_id': q['image_id'],
            'metadata': {'dataset': 'okvqa', 'split': self.split}
        }
    
    def __repr__(self):
        return f"<OKVQADatasetLazy size={len(self)}>"

