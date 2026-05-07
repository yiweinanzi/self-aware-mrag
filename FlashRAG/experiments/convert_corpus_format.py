#!/usr/bin/env python3
"""将MultiModalQA文档转换为FlashRAG格式"""

import json

# 读取原始文档
print("读取原始文档...")
with open('/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/documents.json', 'r') as f:
    documents = []
    for line in f:
        if line.strip():
            doc = json.loads(line)
            documents.append(doc)

print(f"总共读取 {len(documents)} 个文档")

# 转换为FlashRAG格式
print("\n转换为FlashRAG格式...")
output_path = '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/corpus.jsonl'

with open(output_path, 'w') as f:
    for doc in documents:
        # FlashRAG期望的格式：每个文档是一个字典，包含text字段
        new_doc = {
            'title': doc.get('title', ''),
            'text': doc.get('text', ''),
            # 将title和text合并为contents
            'contents': f"{doc.get('title', '')}\n\n{doc.get('text', '')}"
        }

        # 写为一行JSON
        f.write(json.dumps(new_doc) + '\n')

print(f"\n✅ 转换完成！")
print(f"输出文件: {output_path}")
print(f"文档数: {len(documents)}")

# 验证格式
print("\n验证格式（前2行）:")
with open(output_path, 'r') as f:
    for i in range(2):
        line = f.readline()
        if line:
            doc = json.loads(line)
            print(f"\n文档 {i+1}:")
            print(f"  - 有 'title': {bool('title' in doc)}")
            print(f"  - 有 'text': {bool('text' in doc)}")
            print(f"  - 有 'contents': {bool('contents' in doc)}")
            print(f"  - contents长度: {len(doc['contents'])}")