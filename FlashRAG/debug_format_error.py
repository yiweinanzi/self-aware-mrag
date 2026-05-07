#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试格式字符串错误
"""

import os
import sys
import json
from traceback import format_exc

# 添加项目路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_simple_generation():
    """测试最简单的生成"""
    print("测试1: 简单文本生成")
    try:
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper

        # 创建wrapper
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            device='cuda'
        )

        # 简单生成
        text = "What is this?"
        answer = qwen3_vl.generate(
            text=text,
            max_new_tokens=5,
            temperature=0.01
        )

        print(f"✅ 简单生成成功: {answer}")
        return True

    except Exception as e:
        print(f"❌ 简单生成失败: {e}")
        print(format_exc())
        return False

def test_pipeline_generation():
    """测试Pipeline的生成部分"""
    print("\n测试2: Pipeline生成")
    try:
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
        from flashrag.utils.vqa_evaluator import extract_okvqa_answer

        # 创建wrapper
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            device='cuda'
        )

        # 构建prompt
        question = "What sport can you use this for?"
        context = "This is a squash racket."

        prompt = f"""Based on the following evidence, answer the question.

{context}

Question: {question}

Answer with ONLY 1-3 words (all lowercase, no punctuation):"""

        print(f"Prompt:\n{prompt}\n")

        # 生成答案
        answer = qwen3_vl.generate(
            text=prompt,
            max_new_tokens=5,
            temperature=0.01
        )

        print(f"原始答案: {answer!r}")

        # 应用后处理
        processed_answer = extract_okvqa_answer(answer.strip())
        print(f"处理后答案: {processed_answer!r}")

        return True

    except Exception as e:
        print(f"❌ Pipeline生成失败: {e}")
        print(format_exc())
        return False

def test_with_context_formatting():
    """测试带上下文的格式化"""
    print("\n测试3: 上下文格式化")
    try:
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
        from flashrag.utils.vqa_evaluator import extract_okvqa_answer

        # 创建wrapper
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            device='cuda'
        )

        # 模拟Pipeline中的格式化
        retrieved_docs = [
            {
                "contents": "Squash (sport)\nis appropriate for one's skill level.",
                "title": "Squash (sport)",
                "id": 636978
            },
            {
                "contents": "Racket (sports equipment)\nA racket is used for striking a ball in games such as squash.",
                "title": "Racket (sports equipment)",
                "id": 637059
            }
        ]

        # 构建context
        context_parts = ["Retrieved Evidence:"]
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"Document {i}:\n{doc['contents']}"
            )

        context = "\n\n".join(context_parts)
        question = "What sport can you use this for?"

        prompt = f"""Based on the following evidence, answer the question.

{context}

Question: {question}

Answer with ONLY 1-3 words (all lowercase, no punctuation):"""

        print(f"Context:\n{context}\n")
        print(f"Prompt:\n{prompt}\n")

        # 生成答案
        answer = qwen3_vl.generate(
            text=prompt,
            max_new_tokens=5,
            temperature=0.01
        )

        print(f"原始答案: {answer!r}")

        # 应用后处理
        processed_answer = extract_okvqa_answer(answer.strip())
        print(f"处理后答案: {processed_answer!r}")

        return True

    except Exception as e:
        print(f"❌ 上下文格式化失败: {e}")
        print(format_exc())
        return False

def test_dict_formatting():
    """测试字典格式化问题"""
    print("\n测试4: 字典格式化")
    try:
        # 测试可能的字典格式化问题
        test_dict = {"key": "value"}

        # 这些可能导致格式字符串错误
        try:
            result = f"{test_dict}"
            print(f"字典直接格式化: {result}")
        except Exception as e:
            print(f"❌ 字典直接格式化失败: {e}")

        try:
            result = "Value: {key}".format(**test_dict)
            print(f"字典解包格式化: {result}")
        except Exception as e:
            print(f"❌ 字典解包格式化失败: {e}")

        # 测试带有特殊字符的字符串
        test_strings = [
            "squash sport equipment",
            "squash",
            "sport",
            "equipment",
            "",
            " ",
            "\n",
            "\t",
        ]

        for s in test_strings:
            try:
                result = f"Answer: {s}"
                print(f"字符串'{s!r}'格式化成功: {result!r}")
            except Exception as e:
                print(f"❌ 字符串'{s!r}'格式化失败: {e}")

        return True

    except Exception as e:
        print(f"❌ 字典格式化测试失败: {e}")
        print(format_exc())
        return False

def main():
    print("=" * 80)
    print("调试格式字符串错误")
    print("=" * 80)

    results = []

    # 运行各项测试
    results.append(test_simple_generation())
    results.append(test_pipeline_generation())
    results.append(test_with_context_formatting())
    results.append(test_dict_formatting())

    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  简单生成: {'✅' if results[0] else '❌'}")
    print(f"  Pipeline生成: {'✅' if results[1] else '❌'}")
    print(f"  上下文格式化: {'✅' if results[2] else '❌'}")
    print(f"  字典格式化: {'✅' if results[3] else '❌'}")
    print("=" * 80)

if __name__ == "__main__":
    main()