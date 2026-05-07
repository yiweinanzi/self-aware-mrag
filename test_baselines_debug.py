#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script to check why baseline methods have 0% accuracy"""

import json
import os
import glob

def check_method_outputs():
    """Check the outputs of each baseline method"""

    print("="*70)
    print("DEBUG: 检查baseline方法的输出")
    print("="*70)

    print("\n问题分析:")
    print("1. MuRAG/VisRAG: 检索成功但准确率0% -> 答案生成或评估有问题")
    print("2. ViDoRAG: 检索率0% -> 检索器没有返回任何文档")
    print("3. 需要检查答案格式是否正确")
    print("="*70)

    # Check if there are any result files
    result_dir = "/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines"

    # Look for the most recent results
    import glob
    result_files = glob.glob(os.path.join(result_dir, "*results*.json"))
    result_files.sort(key=os.path.getmtime, reverse=True)

    print("\n最近的结果文件:")
    for f in result_files[:5]:
        print(f"  {os.path.basename(f)} ({os.path.getmtime(f):.0f})")

    # Try to load a result file
    if result_files:
        latest_file = result_files[0]
        print(f"\n检查文件: {latest_file}")

        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)

            # Check the structure
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                print(f"结果数量: {len(results)}")

                if results:
                    first_result = results[0]
                    print(f"\n第一个结果的结构:")
                    print(f"  question: {first_result.get('question', 'N/A')[:50]}...")
                    print(f"  answer: {first_result.get('answer', 'N/A')}")
                    print(f"  golden_answers: {first_result.get('golden_answers', 'N/A')}")
                    print(f"  correct: {first_result.get('correct', 'N/A')}")

                    # Check answer format
                    answer = first_result.get('answer', '')
                    golden = first_result.get('golden_answers', [])

                    print(f"\n答案对比:")
                    print(f"  生成的答案: '{answer}' (类型: {type(answer)})")
                    print(f"  标准答案: {golden} (类型: {type(golden)})")

                    if golden and isinstance(golden, list):
                        print(f"  第一个标准答案: '{golden[0]}' (类型: {type(golden[0])})")

                        # Check if answer matches
                        if answer == golden[0]:
                            print("  ✓ 答案匹配!")
                        else:
                            print("  ✗ 答案不匹配")

        except Exception as e:
            print(f"读取结果文件失败: {e}")

if __name__ == "__main__":
    check_method_outputs()