#!/usr/bin/env python3
"""
测试调试版MuRAG - 只运行1个样本
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/data0/home/zqwang/ACL/models/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/data0/home/zqwang/ACL/models/huggingface/transformers'

def main():
    print("🔧 测试调试版MuRAG - 1个样本")
    print("="*60)

    # 导入必要的模块
    from flashrag.wrapper.qwen3vl_wrapper import Qwen3VLWrapper
    from flashrag.retriever.retriever import DenseRetriever
    from FlashRAG.experiments.baselines.murag_enhanced_debug import create_murag_enhanced_debug

    # 初始化模型
    print("\n1. 初始化模型...")
    qwen3vl = Qwen3VLWrapper(
        model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        device='cuda',
        torch_dtype='bfloat16'
    )

    # 初始化检索器
    print("\n2. 初始化��索器...")
    retriever_config = {
        'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
        'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    }

    retriever = DenseRetriever(retriever_config)

    # 创建MuRAG
    print("\n3. 创建MuRAG...")
    config = {
        'retrieval_topk': 5,
        'ensemble_k': 3,
        'temperature': 0.01,
    }

    murag = create_murag_enhanced_debug(qwen3vl, retriever, config)

    # 准备测试样本
    print("\n4. 准备测试样本...")
    sample = {
        'question': 'What sport can you use this for?',
        'golden_answers': ['race', 'race', 'race'],
        'image': None  # 暂时不使用图像
    }

    # 运行测试
    print("\n5. 运行MuRAG测试...")
    try:
        result = murag.run_single(sample)

        print("\n" + "="*60)
        print("📊 测试结果:")
        print(f"问题: {result['question']}")
        print(f"生成答案: '{result['answer']}'")
        print(f"正确答案: {sample['golden_answers']}")
        print(f"是否正确: {'✅' if result.get('is_correct', False) else '❌'}")
        print(f"检索文档数: {len(result.get('retrieved_docs', []))}")
        if 'sub_answers' in result:
            print(f"子答案: {result['sub_answers']}")

        print("\n" + "="*60)
        print("✅ 测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()