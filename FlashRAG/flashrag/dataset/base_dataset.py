# -*- coding: utf-8 -*-
"""
BaseDataset 模板
所有自定义数据集（如 OK-VQA、WebQA 等）都应继承此类
"""

import os
import json
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    FlashRAG 数据集基类
    所有数据集需继承此类，并实现 load_data()
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config.get('name', 'base')
        self.data = self.load_data()
    
    def load_data(self):
        """
        子类必须实现：用于加载原始数据并返回列表
        每个元素必须是一个 dict，包含至少以下字段：
        {
            'id': 样本唯一ID,
            'question': 问题文本,
            'golden_answers': 答案（list[str]）,
            'context': （可选）检索上下文,
            'image': （可选）图像对象或路径
        }
        """
        raise NotImplementedError("Subclasses must implement load_data().")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} size={len(self)}>"
