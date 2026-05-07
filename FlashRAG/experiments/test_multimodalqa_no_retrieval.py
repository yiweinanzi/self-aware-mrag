#!/usr/bin/env python3
"""测试MultiModalQA不使用检索的效果"""

import sys
import json
import gzip
from datetime import datetime

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

print("=" * 80)
print("MultiModalQA无检索测试")
print("=" * 80)
print(f"开始时间: {datetime.now()}")
print()

# 加载数据
print("1. 加载数据...")
data_file = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_dev.jsonl.gz'
samples = []

# 加载相关文档
print("2. 加载相关文档...")
texts = {}
with gzip.open('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_texts.jsonl.gz', 'rt') as f:
    for line in f:
        item = json.loads(line)
        texts[item['id']] = item['text']

tables = {}
with gzip.open('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_tables.jsonl.gz', 'rt') as f:
    for line in f:
        item = json.loads(line)
        tables[item['id']] = item['table']

# 处理前3个样本
with gzip.open(data_file, 'rt') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        item = json.loads(line)

        metadata = item.get('metadata', {})
        question = item.get('question', '')
        text_doc_ids = metadata.get('text_doc_ids', [])
        table_id = metadata.get('table_id', '')
        image_doc_ids = metadata.get('image_doc_ids', [])

        # 获取答案
        answers = item.get('answers', [])
        golden_answers = [ans.get('answer', '') for ans in answers]

        # 构建上下文
        context = []
        if text_doc_ids:
            context.append(f"相关文本: {texts[text_doc_ids[0]][:200]}...")
        if table_id and table_id in tables:
            # 表格是字典结构，取第一行作为示例
            table = tables[table_id]
            if 'table_rows' in table and table['table_rows']:
                first_row = table['table_rows'][0]
                table_text = ' | '.join([cell.get('text', '') for cell in first_row[:3]])
                context.append(f"相关表格: {table_text}...")
        if image_doc_ids:
            context.append(f"相关图像: {len(image_doc_ids)}张")

        print(f"\n样本 {i+1}:")
        print(f"问题: {question}")
        print(f"答案类型: {metadata.get('type', 'Unknown')}")
        print(f"文本文档数: {len(text_doc_ids)}")
        print(f"有表格: {'是' if table_id else '否'}")
        print(f"图像数: {len(image_doc_ids)}")
        print(f"上下文: {context[0] if context else '无'}")
        print(f"黄金答案: {golden_answers}")

        samples.append({
            'id': item.get('qid', f'mmqa_{i}'),
            'question': question,
            'golden_answers': golden_answers,
            'context': context,
            'text_doc_ids': text_doc_ids,
            'table_id': table_id,
            'image_doc_ids': image_doc_ids
        })

print(f"\n✅ 加载成功: {len(samples)} 样本")

# 总结
print("\n" + "=" * 80)
print("总结:")
print("MultiModalQA是给定文档的问答任务，不是检索任务")
print("每个样本都提供了相关的文档ID:")
print("- text_doc_ids: 相关文本文档")
print("- table_id: 相关表格")
print("- image_doc_ids: 相关图像")
print("=" * 80)
print(f"完成时间: {datetime.now()}")