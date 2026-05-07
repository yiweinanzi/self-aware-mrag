#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试Self-RAG集成是否正常
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_selfrag_import():
    """测试Self-RAG模块导入"""
    print("=" * 60)
    print("测试1: Self-RAG模块导入")
    print("=" * 60)
    
    try:
        from flashrag.baseline.selfrag import SelfRAG, CONTROL_TOKENS
        print("✅ Self-RAG模块导入成功")
        print(f"   - Control tokens: {len(CONTROL_TOKENS)} 个")
        print(f"   - 示例tokens: {CONTROL_TOKENS[:3]}")
        return True
    except Exception as e:
        print(f"❌ Self-RAG模块导入失败: {e}")
        return False

def test_selfrag_initialization():
    """测试Self-RAG初始化"""
    print("\n" + "=" * 60)
    print("测试2: Self-RAG初始化")
    print("=" * 60)
    
    try:
        from flashrag.baseline.selfrag import SelfRAG
        
        # 测试初始化（不实际加载模型）
        selfrag = SelfRAG(
            model_name="selfrag/selfrag_llama2_7b",
            threshold=0.2,
            max_new_tokens=100
        )
        
        print("✅ Self-RAG对象创建成功")
        print(f"   - 模型名: {selfrag.model_name}")
        print(f"   - 检索阈值: {selfrag.threshold}")
        print(f"   - 最大token数: {selfrag.max_new_tokens}")
        print(f"   - 权重: w_rel={selfrag.w_rel}, w_sup={selfrag.w_sup}, w_use={selfrag.w_use}")
        
        return True
    except Exception as e:
        print(f"❌ Self-RAG初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_format_prompt():
    """测试prompt格式化"""
    print("\n" + "=" * 60)
    print("测试3: Prompt格式化")
    print("=" * 60)
    
    try:
        from flashrag.baseline.selfrag import SelfRAG
        
        selfrag = SelfRAG()
        
        # 测试无context的prompt
        question = "What is the capital of France?"
        prompt1 = selfrag.format_prompt(question)
        print("✅ 无context prompt:")
        print(f"   {prompt1[:100]}...")
        
        # 测试有context的prompt
        context = "Paris is the capital and most populous city of France."
        prompt2 = selfrag.format_prompt(question, context)
        print("\n✅ 有context prompt:")
        print(f"   {prompt2[:150]}...")
        
        return True
    except Exception as e:
        print(f"❌ Prompt格式化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_script():
    """测试对比脚本能否正常导入"""
    print("\n" + "=" * 60)
    print("测试4: 对比脚本导入")
    print("=" * 60)
    
    try:
        # 检查文件是否存在
        script_path = 'experiments/all_methods_comparison_with_selfrag.py'
        if not os.path.exists(script_path):
            print(f"❌ 脚本文件不存在: {script_path}")
            return False
        
        print(f"✅ 对比脚本存在: {script_path}")
        
        # 读取脚本检查关键函数
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        functions = ['run_selfrag', 'run_murag', 'run_mr2ag', 'run_visrag', 
                    'run_reveal', 'run_ragvl', 'run_ours']
        
        all_found = True
        for func in functions:
            if f"def {func}(" in content:
                print(f"   ✅ 找到函数: {func}")
            else:
                print(f"   ❌ 缺失函数: {func}")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"❌ 对比脚本检查失败: {e}")
        return False

def test_selfrag_readme():
    """测试Self-RAG README是否存在"""
    print("\n" + "=" * 60)
    print("测试5: Self-RAG文档")
    print("=" * 60)
    
    try:
        readme_path = 'flashrag/baseline/SELFRAG_README.md'
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✅ Self-RAG README存在")
            print(f"   - 文件大小: {len(content)} 字符")
            print(f"   - 包含'官方代码': {'官方代码' in content}")
            print(f"   - 包含'核心创新': {'核心创新' in content}")
            print(f"   - 包含'运行实验': {'运行实验' in content}")
            return True
        else:
            print(f"❌ README不存在: {readme_path}")
            return False
    except Exception as e:
        print(f"❌ README检查失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("Self-RAG 集成测试")
    print("🧪" * 30 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_selfrag_import()))
    results.append(("对象初始化", test_selfrag_initialization()))
    results.append(("Prompt格式化", test_format_prompt()))
    results.append(("对比脚本", test_comparison_script()))
    results.append(("README文档", test_selfrag_readme()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！Self-RAG集成成功！")
        print("\n📝 下一步:")
        print("   1. 运行完整对比实验:")
        print("      python experiments/all_methods_comparison_with_selfrag.py --max_samples 100")
        print("\n   2. 查看Self-RAG文档:")
        print("      cat flashrag/baseline/SELFRAG_README.md")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查上述错误信息")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()

