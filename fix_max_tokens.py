#!/usr/bin/env python3
"""
修复max_new_tokens问题
"""

import os
import re

# 需要修复的文件
files_to_fix = [
    '/data0/home/zqwang/ACL/FlashRAG/experiments/baselines/murag_enhanced.py',
    '/data0/home/zqwang/ACL/FlashRAG/experiments/baselines/visrag_enhanced.py',
    '/data0/home/zqwang/ACL/FlashRAG/experiments/baselines/vidorag_pipeline.py',
    '/data0/home/zqwang/ACL/FlashRAG/experiments/baselines/ragvl_enhanced.py',
    '/data0/home/zqwang/ACL/FlashRAG/experiments/baselines/samrag_adapted.py',
    '/data0/home/zqwang/ACL/FlashRAG/experiments/baselines/mr2ag_enhanced.py',
]

print("修复max_new_tokens问题...")
print("="*60)

for file_path in files_to_fix:
    filename = os.path.basename(file_path)
    print(f"\n修复 {filename}...")

    # 读取文件
    with open(file_path, 'r') as f:
        content = f.read()

    # 统计需要替换的地方
    old_pattern = 'max_new_tokens=10'
    count = content.count(old_pattern)

    if count > 0:
        # 替换为max_new_tokens=20
        content = content.replace('max_new_tokens=10', 'max_new_tokens=20')

        # 写回文件
        with open(file_path, 'w') as f:
            f.write(content)

        print(f"  ✓ 替换了 {count} 处 max_new_tokens=10 -> max_new_tokens=20")
    else:
        print(f"  - 没有找到需要替换的地方")

print("\n" + "="*60)
print("✅ 修复完成！")
print("\n问题分析：")
print("1. max_new_tokens=10 太小，无法生成完整答案")
print("2. 已修改为 max_new_tokens=20，足够生成1-3个单词的答案")
print("3. 这将提高准确率，因为模型可以生成完整的答案")
print("="*60)