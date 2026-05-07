"""
CLIP图像检索器
支持真正的以图搜图功能
"""

import warnings
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Dict
import json
import os

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CLIPImageRetriever:
    """
    CLIP图像检索器
    用于基于图像查询检索相关的文本描述
    """

    def __init__(self, model_path: str, corpus_path: str, device: str = 'cuda'):
        """
        初始化CLIP图像检索器

        Args:
            model_path: CLIP模型路径
            corpus_path: 图像语料库路径（jsonl文件）
            device: 设备
        """
        if not CLIP_AVAILABLE:
            raise ImportError("需要安装transformers库")

        self.device = device
        self.model_path = model_path
        self.corpus_path = corpus_path

        # 加载CLIP模型
        print(f"正在加载CLIP模型: {model_path}")
        self.clip_model = CLIPModel.from_pretrained(model_path).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_path)
        self.clip_model.eval()
        print("✅ CLIP模型加载成功")

        # 加载语料库
        self.load_corpus()

        # 预计算语料库的文本特征
        self.precompute_text_features()

    def load_corpus(self):
        """加载图像语料库"""
        print(f"正在加载图像语料库: {self.corpus_path}")

        self.corpus = []
        with open(self.corpus_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.corpus.append(item)

        print(f"✅ 加载了 {len(self.corpus)} 个图像")

    def precompute_text_features(self):
        """预计算语料库的文本特征"""
        print("正在预计算文本特征...")

        texts = [item['text'] for item in self.corpus]
        batch_size = 32
        all_features = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = self.clip_processor(
                text=batch,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                batch_features = self.clip_model.get_text_features(**inputs)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                all_features.append(batch_features.cpu())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  已处理 {min(i + batch_size, len(texts))}/{len(texts)} 条")

        # 合并所有特征
        self.text_features = torch.cat(all_features, dim=0).to(self.device)
        print(f"✅ 文本特征计算完成: {self.text_features.shape}")

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """编码单个图像"""
        inputs = self.clip_processor(
            images=[image],
            return_tensors='pt',
            padding=True
        ).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def retrieve(self, query_image: Image.Image, top_k: int = 5) -> Tuple[List[Dict], List[float]]:
        """
        基于图像检索相关的文本描述

        Args:
            query_image: 查询图像
            top_k: 返回结果数量

        Returns:
            (results, scores)
        """
        # 编码查询图像
        query_features = self.encode_image(query_image)

        # 计算相似度
        similarities = torch.matmul(query_features, self.text_features.T).squeeze(0)

        # 获取top-k结果
        top_k = min(top_k, len(self.corpus))
        top_scores, top_indices = torch.topk(similarities, k=top_k)

        # 准备结果
        results = []
        scores_list = top_scores.cpu().numpy().tolist()

        for idx, score in zip(top_indices.cpu().numpy(), scores_list):
            item = self.corpus[idx]
            # 转换为FlashRAG期望的格式
            result = {
                'docid': item['docid'],
                'contents': item['text'],
                'title': item['title'],
                'image_path': item.get('image_path', ''),
                'scenario': item.get('scenario', ''),
            }
            results.append(result)

        return results, scores_list


if __name__ == '__main__':
    # 测试代码
    import sys
    sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

    # 创建测试图像
    from PIL import Image
    import numpy as np

    # 加载一个测试图像
    test_image_path = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/image_corpus/Mechanical_Can_7_gt_hq2.jpg'
    if os.path.exists(test_image_path):
        test_image = Image.open(test_image_path).convert('RGB')

        # 初始化检索器
        retriever = CLIPImageRetriever(
            model_path='/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
            corpus_path='/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/image_corpus.jsonl',
            device='cuda'
        )

        # 测试检索
        results, scores = retriever.retrieve(test_image, top_k=5)

        print("\n检索结果：")
        for i, (result, score) in enumerate(zip(results, scores)):
            print(f"{i+1}. [Score: {score:.4f}] {result['title']}")
            print(f"   {result['contents'][:100]}...")
    else:
        print(f"测试图像不存在: {test_image_path}")