#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细调试生成答案的错误
"""

import os
import sys
import json
import traceback

# 添加项目路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def debug_single_sample():
    """调试单个样本的完整流程"""
    print("=" * 80)
    print("调试单个样本的完整流程")
    print("=" * 80)

    try:
        # 1. 加载模型
        print("\n1. 初始化模型")
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
        from flashrag.retriever import DenseRetriever
        from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL

        # 创建模型
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            device='cuda'
        )
        print("✅ Qwen3-VL创建成功")

        # 创建检索器
        retriever = DenseRetriever(
            faiss_index_path='/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
            corpus_path='/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
            retrieval_model_path='/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
            max_length=512
        )
        print("✅ 检索器创建成功")

        # 创建Pipeline
        pipeline = SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config={
                'force_retrieval': True,
                'uncertainty_threshold': 0.43,
                'use_improved_estimator': True,
                'use_position_fusion': True,
                'use_attribution': True
            }
        )
        print("✅ Pipeline创建成功")

        # 2. 加载数据
        print("\n2. 加载数据")
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

        dataset = OKVQADatasetSimple({
            'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
            'split': 'val',
            'load_images': False,
        })
        print(f"✅ 数据集加载成功，共{len(dataset)}个样本")

        # 获取第一个样本
        sample = dataset[0]
        print(f"✅ 获取第一个样本")
        print(f"   问题: {sample['question']}")
        print(f"   答案: {sample['answer']}")

        # 3. 逐步调试Pipeline流程
        print("\n3. 逐步调试Pipeline流程")

        # 3.1 不确定性估计
        question = sample['question']
        print(f"\n[步骤1] 不确定性估计")
        print(f"问题: {question[:50]}...")

        uncertainty = pipeline.uncertainty_estimator.estimate(
            text=question,
            image=None
        )
        print(f"不确定性估计结果: {uncertainty}")

        # 3.2 检索
        print(f"\n[步骤2] 检索")
        if hasattr(pipeline.retriever, 'search'):
            search_results = pipeline.retriever.search(question, num=5, return_score=True)
            if isinstance(search_results, tuple):
                retrieved_docs, retrieval_scores = search_results
            else:
                retrieved_docs = search_results
                retrieval_scores = [1.0] * len(retrieved_docs) if retrieved_docs else []

            print(f"检索到{len(retrieved_docs)}个文档")
            for i, doc in enumerate(retrieved_docs[:2]):
                print(f"  文档{i+1}: {doc.get('title', 'N/A')[:50]}...")

        # 3.3 构建context
        print(f"\n[步骤3] 构建context")
        context_parts = ["Retrieved Evidence:"]
        for i, doc in enumerate(retrieved_docs[:3], 1):
            doc_text = doc.get('contents', '') if isinstance(doc, dict) else str(doc)
            doc_text = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            context_parts.append(
                f"Document {i}:\n{doc_text}"
            )

        context = "\n\n".join(context_parts)
        print(f"Context长度: {len(context)}字符")

        # 3.4 生成答案 - 这里详细调试
        print(f"\n[步骤4] 生成答案")
        try:
            # 构建prompt
            prompt = f"""Based on the following evidence, answer the question.

{context}

Question: {question}

Answer with ONLY 1-3 words (all lowercase, no punctuation):"""

            print(f"Prompt构建成功，长度: {len(prompt)}")
            print(f"Prompt前100字符: {prompt[:100]}...")

            # 生成答案
            print("调用qwen3_vl.generate...")
            answer = qwen3_vl.generate(
                text=prompt,
                max_new_tokens=5,
                temperature=0.01
            )
            print(f"✅ 生成成功，原始答案: {answer!r}")

            # 后处理
            from flashrag.utils.vqa_evaluator import extract_okvqa_answer
            processed_answer = extract_okvqa_answer(answer.strip())
            print(f"✅ 后处理成功，最终答案: {processed_answer!r}")

        except Exception as e:
            print(f"❌ 生成答案失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            return False

        # 3.5 运行完整的run_single
        print(f"\n[步骤5] 运行完整的run_single")
        try:
            result = pipeline.run_single(sample)
            print(f"✅ run_single成功")
            print(f"   问题: {result.get('question', 'N/A')[:50]}...")
            print(f"   答案: {result.get('answer', 'N/A')!r}")
            print(f"   检索: {result.get('retrieved', False)}")

        except Exception as e:
            print(f"❌ run_single失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            return False

        return True

    except Exception as e:
        print(f"❌ 调试失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def test_json_serialization():
    """测试JSON序列化问题"""
    print("\n" + "=" * 80)
    print("测试JSON序列化问题")
    print("=" * 80)

    # 测试可能的序列化问题
    test_cases = [
        # 基本类型
        {"key": "value"},
        {"answer": "squash"},
        # 包含None的字典
        {"answer": None},
        {"answer": ""},
        # 嵌套字典
        {"uncertainty": {"text": 0.39, "visual": 0.0, "total": 0.195}},
        # 包含特殊字符
        {"answer": "squash sport equipment"},
        {"answer": "sport\tequipment"},
        # 包含列表
        {"golden_answers": ["race", "race", "race"]},
        # 复杂结构
        {
            "question": "What sport can you use this for?",
            "answer": "",
            "uncertainty": {"total": 0.195, "text": 0.39, "visual": 0.0},
            "retrieved": True,
            "retrieved_docs": [],
            "golden_answers": ["race", "race", "race"]
        }
    ]

    for i, test_dict in enumerate(test_cases, 1):
        try:
            json_str = json.dumps(test_dict)
            print(f"测试{i}: ✅ 成功序列化")
        except Exception as e:
            print(f"测试{i}: ❌ 序列化失败: {e}")
            print(f"  字典内容: {test_dict}")

if __name__ == "__main__":
    success = debug_single_sample()
    test_json_serialization()

    print("\n" + "=" * 80)
    if success:
        print("✅ 调试成功完成")
    else:
        print("❌ 调试过程中发现问题")
    print("=" * 80)