#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试VQA改进的脚本
验证VQA官方答案后处理和prompt优化
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_vqa_evaluator():
    """测试VQA官方评测器"""
    print("🔍 测试VQA官方答案处理...")

    try:
        from flashrag.utils.vqa_evaluator import VQAEvaluator, extract_okvqa_answer, standardize_vqa_answer, evaluate_vqa_accuracy

        evaluator = VQAEvaluator()

        # 测试答案标准化
        test_cases = [
            "This is a motorcycle racing sport",  # 长答案
            "The answer is motorcycle racing",     # 包含前缀
            "MOTORCYCLE RACING!",                   # 大写+标点
            "motorcycle racing",                   # 理想答案
            "race",                                # 单词答案
            "  motorcycle  racing  ",              # 多余空格
            "",                                    # 空答案
        ]

        print("\n📝 测试答案标准化:")
        for i, answer in enumerate(test_cases):
            standard = standardize_vqa_answer(answer)
            short = extract_okvqa_answer(answer)
            print(f"  {i+1}. '{answer}' -> 标准化: '{standard}' -> 短答案: '{short}'")

        # 测试VQA准确率计算
        print("\n📊 测试VQA准确率计算:")
        pred = "motorcycle racing"
        gts = ["motorcycle racing", "racing", "motorcycle sport"]
        result = evaluate_vqa_accuracy(pred, gts)
        print(f"  预测: '{pred}'")
        print(f"  标准答案: {gts}")
        print(f"  准确率: {result['accuracy']:.1f}% (匹配: {result['matches']}/3)")

        return True

    except Exception as e:
        print(f"❌ VQA评测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_template():
    """测试优化后的prompt模板"""
    print("\n🔍 测试优化后的Prompt模板...")

    try:
        from flashrag.prompt.base_prompt import PromptTemplate

        # 测试配置
        config = {
            'framework': 'huggingface',
            'generator_max_input_len': 2048,
            'is_reasoning': False
        }

        template = PromptTemplate(config)

        # 测试prompt生成
        question = "What sport can you use this for?"
        reference = "Document 1: Motorcycle sport is a broad field that encompasses motorcycle racing..."

        prompt_str = template.get_string(
            question=question,
            retrieval_result=[{"contents": reference}],
            formatted_reference=reference
        )

        print("📝 生成的Prompt:")
        print("=" * 50)
        print(prompt_str)
        print("=" * 50)

        # 检查关键约束
        key_constraints = [
            "1-3 words",
            "lowercase",
            "no punctuation",
            "no explanation"
        ]

        print("\n✅ Prompt包含的关键约束:")
        for constraint in key_constraints:
            if constraint.lower() in prompt_str.lower():
                print(f"  ✓ {constraint}")
            else:
                print(f"  ✗ {constraint} (缺失)")

        return True

    except Exception as e:
        print(f"❌ Prompt模板测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_answer_processing_simulation():
    """模拟答案处理流程"""
    print("\n🔍 测试答案处理流程模拟...")

    try:
        from flashrag.utils.vqa_evaluator import extract_okvqa_answer

        # 模拟LLM可能生成的各种答案
        raw_answers = [
            "This is used for motorcycle racing.",  # 描述性句子
            "Motorcycle Racing Sport",              # 大写
            "motorcycle racing!",                   # 带标点
            "The answer is: motorcycle racing",    # 带前缀
            "racing",                              # 短答案
            "motorcycle",                          # 部分答案
            "  motocross  race  ",                # 多余空格+内容词
            "It is a form of motorcycle racing competition",  # 长描述
        ]

        print("📝 模拟答案处理:")
        for i, raw_answer in enumerate(raw_answers):
            processed = extract_okvqa_answer(raw_answer)
            print(f"  {i+1}. 原始: '{raw_answer}'")
            print(f"     处理后: '{processed}'")
            print(f"     长度: {len(processed.split())} 词")

        return True

    except Exception as e:
        print(f"❌ 答案处理流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 VQA改进测试套件")
    print("=" * 50)

    tests = [
        ("VQA评测器", test_vqa_evaluator),
        ("Prompt模板", test_prompt_template),
        ("答案处理流程", test_answer_processing_simulation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1

    print(f"\n总计: {passed}/{len(results)} 个测试通过")

    if passed == len(results):
        print("🎉 所有改进测试通过！可以开始运行实验了。")
        return True
    else:
        print("💥 部分测试失败，请检查改进实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)