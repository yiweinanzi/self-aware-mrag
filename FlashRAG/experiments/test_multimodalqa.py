#!/usr/bin/env python3
"""测试MultiModalQA数据加载和处理"""

import sys
import json
import gzip
from datetime import datetime

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

print("="*80)
print("MultiModalQA数据测试")
print("="*80)
print(f"开始时间: {datetime.now()}")
print()

# 加载数据
print("1. 加载数据...")
data_file = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_dev.jsonl.gz'
samples = []

with gzip.open(data_file, 'rt') as f:
    for i, line in enumerate(f):
        if i >= 5:  # 只加载5个样本
            break
        item = json.loads(line.strip())

        # 获取元数据
        metadata = item.get('metadata', {})
        question_type = metadata.get('type', 'Unknown')
        modalities = metadata.get('modalities', [])
        image_doc_ids = metadata.get('image_doc_ids', [])
        text_doc_ids = metadata.get('text_doc_ids', [])
        table_id = metadata.get('table_id', '')

        # 获取答案
        answers = item.get('answers', [])
        golden_answers = [ans.get('answer', '') for ans in answers]
        answer_type = answers[0].get('type', 'string') if answers else 'string'
        modality = answers[0].get('modality', '') if answers else ''

        # 构建样本
        sample = {
            'id': item.get('qid', f'mmqa_{i}'),
            'question': item.get('question', ''),
            'golden_answers': golden_answers,
            'answer': golden_answers[0] if golden_answers else '',
            'question_type': question_type,
            'modalities': modalities,
            'answer_type': answer_type,
            'answer_modality': modality,
            'metadata': metadata,
            'image_doc_ids': image_doc_ids,
            'text_doc_ids': text_doc_ids,
            'table_id': table_id,
        }

        # 处理图像
        if image_doc_ids:
            print(f"\n样本 {i+1}:")
            print(f"  问题类型: {question_type}")
            print(f"  模态: {modalities}")
            print(f"  图像文档数: {len(image_doc_ids)}")
            print(f"  文本文档数: {len(text_doc_ids)}")
            print(f"  答案: {golden_answers}")

            # 检查图像文件
            for img_id in image_doc_ids[:2]:  # 只检查前2个
                img_path = f"/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/images/final_dataset_images/{img_id}.jpg"
                import os
                if os.path.exists(img_path):
                    sample['image'] = img_path
                    print(f"  ✓ 找到图像: {img_id} -> {img_path}")
                else:
                    print(f"  ✗ 图像不存在: {img_id} -> {img_path}")

        samples.append(sample)

print(f"\n✅ 加载成功: {len(samples)} 样本")

# 打印统计
print("\n2. 数据统计:")
from collections import Counter
type_counter = Counter([s['question_type'] for s in samples])
modality_counter = Counter([','.join(s['modalities']) for s in samples if s['modalities']])

print(f"\n问题类型分布:")
for qtype, count in type_counter.items():
    print(f"  - {qtype}: {count} 样本")

print(f"\n模态分布:")
for mod, count in modality_counter.items():
    print(f"  - {mod}: {count} 样本")

# 检查图像路径
print("\n3. 图像检查:")
has_image = 0
for i, s in enumerate(samples):
    if s.get('image'):
        has_image += 1
        print(f"  样本 {i+1}: {s['image']}")
    else:
        print(f"  样本 {i+1}: 无图像")

print(f"\n有图像的样本: {has_image}/{len(samples)}")

# 展示第一个样本的详细信息
if samples:
    print("\n4. 样本详情 (第一个):")
    sample = samples[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))

print("\n" + "="*80)
print(f"完成时间: {datetime.now()}")