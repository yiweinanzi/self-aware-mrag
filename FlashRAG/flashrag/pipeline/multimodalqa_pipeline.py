"""MultiModalQA专用Pipeline，使用数据集提供的文档而不进行检索"""

import json
import gzip
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from ..modules.qwen3_vl import create_qwen3_vl_wrapper
from ..utils import get_dataset


class MultiModalQAPipeline:
    """MultiModalQA专用Pipeline，直接使用数据集中提供的文档"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        dataset_path: str = None,
        thinking: bool = False,
        max_images: int = 20,
    ):
        """
        初始化MultiModalQA Pipeline

        Args:
            model_path: Qwen3-VL模型路径
            device: 设备
            torch_dtype: 数据类型
            dataset_path: MultiModalQA数据集路径
            thinking: 是否启用thinking模式
            max_images: 最大图像数量
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.dataset_path = dataset_path or "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA"
        self.thinking = thinking
        self.max_images = max_images

        # 初始化模型
        print("初始化Qwen3-VL模型...")
        self.qwen3_vl = create_qwen3_vl_wrapper(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype
        )

        # 加载MultiModalQA文档集合
        self._load_multimodalqa_documents()

        print(f"✅ MultiModalQAPipeline初始化完成")
        print(f"  - 已加载 {len(self.texts)} 个文本文档")
        print(f"  - 已加载 {len(self.tables)} 个表格")
        print(f"  - 图像目录: {self.image_dir}")

    def _load_multimodalqa_documents(self):
        """加载MultiModalQA的所有文档"""
        print("加载MultiModalQA文档集合...")

        # 加载文本
        self.texts = {}
        with gzip.open(os.path.join(self.dataset_path, "MMQA_texts.jsonl.gz"), 'rt') as f:
            for line in f:
                item = json.loads(line)
                self.texts[item['id']] = item

        # 加载表格
        self.tables = {}
        with gzip.open(os.path.join(self.dataset_path, "MMQA_tables.jsonl.gz"), 'rt') as f:
            for line in f:
                item = json.loads(line)
                self.tables[item['id']] = item

        # 设置图像目录
        self.image_dir = os.path.join(self.dataset_path, "images/final_dataset_images")

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本

        Args:
            sample: 包含question、metadata等信息的样本

        Returns:
            处理结果
        """
        question = sample.get('question', '')
        metadata = sample.get('metadata', {})

        # 提取相关文档
        text_doc_ids = metadata.get('text_doc_ids', [])
        table_id = metadata.get('table_id', '')
        image_doc_ids = metadata.get('image_doc_ids', [])

        # 构建上下文
        context_docs = []

        # 添加文本上下文
        for doc_id in text_doc_ids:
            if doc_id in self.texts:
                doc = self.texts[doc_id]
                context_docs.append({
                    'id': doc_id,
                    'contents': doc['text'],
                    'title': doc.get('title', ''),
                    'source': 'multimodalqa_text'
                })

        # 添加表格上下文
        if table_id and table_id in self.tables:
            table = self.tables[table_id]
            # 将表格转换为文本格式
            table_text = self._table_to_text(table)
            context_docs.append({
                'id': table_id,
                'contents': table_text,
                'title': table.get('title', ''),
                'source': 'multimodalqa_table'
            })

        # 获取图像路径
        image_paths = []
        for img_id in image_doc_ids:
            # 尝试不同的扩展名
            for ext in ['.jpg', '.JPG', '.png']:
                img_path = os.path.join(self.image_dir, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    break

        # 使用图像（如果有）
        image = image_paths[0] if image_paths else None

        # 直接使用Qwen3-VL生成答案（不进行检索）
        start_time = datetime.now()

        # 构建prompt，包含上下文信息
        context_prompt = ""
        if context_docs:
            context_prompt = "\n\n相关文档：\n"
            for i, doc in enumerate(context_docs[:3]):  # 只使用前3个文档避免太长
                context_prompt += f"\n文档{i+1}（来源：{doc['source']}）：\n{doc['contents'][:500]}...\n"

        full_prompt = f"{context_prompt}\n\n问题：{question}\n\n请根据上述文档回答问题。"

        # 生成答案
        answer = self.qwen3_vl.generate_answer(
            prompt=full_prompt,
            image=image
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            'answer': answer,
            'question': question,
            'golden_answers': sample.get('golden_answers', []),
            'retrieved': True,  # 使用了提供的文档
            'retrieved_docs': context_docs,
            'n_retrieved_docs': len(context_docs),
            'image_paths': image_paths,
            'n_images': len(image_paths),
            'processing_time': processing_time,
            'metadata': metadata
        }

    def _table_to_text(self, table: Dict) -> str:
        """将表格转换为文本格式"""
        if not table or 'table_rows' not in table:
            return ""

        rows = table['table_rows']
        if not rows:
            return ""

        # 获取表头（如果有）
        header = table.get('header', [])
        header_text = []
        if header:
            header_text = [h.get('column_name', '') for h in header if 'column_name' in h]

        # 转换前几行数据
        table_text_lines = []

        # 添加表头
        if header_text:
            table_text_lines.append(" | ".join(header_text[:5]))  # 只显示前5列

        # 添加数据行（最多5行）
        for row in rows[:5]:
            row_text = []
            for cell in row[:5]:  # 只显示前5列
                if isinstance(cell, dict):
                    row_text.append(cell.get('text', ''))
                else:
                    row_text.append(str(cell))
            table_text_lines.append(" | ".join(row_text))

        return "\n".join(table_text_lines)


# 便捷函数
def create_multimodalqa_pipeline(
    model_path: str,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    dataset_path: str = None,
    **kwargs
) -> MultiModalQAPipeline:
    """创建MultiModalQA Pipeline实例"""
    return MultiModalQAPipeline(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
        dataset_path=dataset_path,
        **kwargs
    )