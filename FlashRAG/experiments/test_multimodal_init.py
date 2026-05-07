#!/usr/bin/env python3
"""
测试多模态检索器初始化
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import os

def test_multimodal_retriever():
    """测试多模态检索器初始化"""
    print("="*80)
    print("测试多模态检索器初始化")
    print("="*80)

    # 配置
    config = {
        'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
        'retrieval_method': 'e5',
        'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
        'retrieval_topk': 5,
        'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench',
        'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
        'use_multimodal_retrieval': True,
    }

    # 检查CLIP索引路径
    print("\n1. 检查CLIP索引路径...")
    clip_index_dir = config.get('clip_index_path')
    possible_names = ['clip_image_Flat.index', 'clip_Flat.index']
    clip_index_file = None

    print(f"   搜索目录: {clip_index_dir}")
    print(f"   查找文件: {possible_names}")

    for name in possible_names:
        candidate = os.path.join(clip_index_dir, name)
        print(f"   检查: {candidate}")
        if os.path.exists(candidate):
            clip_index_file = candidate
            print(f"   ✓ 找到: {clip_index_file}")
            break

    if clip_index_file is None:
        print("   ✗ 没有找到CLIP索引文件")
        return False

    # 检查图像语料库
    print("\n2. 检查图像语料库...")
    image_corpus_path = os.path.join(clip_index_dir, 'image_corpus.jsonl')
    print(f"   路径: {image_corpus_path}")
    if os.path.exists(image_corpus_path):
        print(f"   ✓ 图像语料库存在")
    else:
        print(f"   ✗ 图像语料库不存在")

    # 初始化多模态检索器
    print("\n3. 初始化多模态检索器...")
    try:
        from flashrag.retriever import DenseRetriever
        from flashrag.retriever.multimodal_retriever import SelfAwareMultimodalRetriever

        # 初始化BGE文本检索器
        print("   初始化BGE文本检索器...")
        bge_retriever_config = {
            'index_path': config['index_path'],
            'corpus_path': config['corpus_path'],
            'retrieval_method': 'e5',
            'retrieval_model_path': config['retrieval_model_path'],
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'cls',
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 128,
            'retrieval_topk': config['retrieval_topk'],
            'save_retrieval_cache': False,
            'use_retrieval_cache': False,
            'retrieval_cache_path': None,
            'use_reranker': False,
            'device': 'cuda',
            'use_sentence_transformer': False,
            'faiss_gpu': False,
            'instruction': '',
        }

        bge_retriever = DenseRetriever(bge_retriever_config)
        print("   ✓ BGE检索器加载成功")

        # 初始化CLIP视觉检索器
        print("   初始化CLIP视觉检索器...")
        clip_retriever_config = {
            'index_path': clip_index_file,
            'corpus_path': config['corpus_path'],
            'retrieval_method': 'clip',
            'retrieval_model_path': config.get('clip_model_path'),
            'retrieval_query_max_length': 77,
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 64,
            'retrieval_topk': config['retrieval_topk'],
            'save_retrieval_cache': False,
            'use_retrieval_cache': False,
            'retrieval_cache_path': None,
            'index_modal': 'all',
            'use_reranker': False,
            'device': 'cuda',
            'retrieval_pooling_method': 'mean',
            'use_sentence_transformer': False,
            'faiss_gpu': False,
            'instruction': '',
        }

        clip_retriever = DenseRetriever(clip_retriever_config)
        print("   ✓ CLIP视觉检索器加载成功")

        # 创建多模态融合检索器
        print("   创建多模态融合检索器...")
        multimodal_config = {
            'retrieval_topk': config['retrieval_topk'],
            'use_clip': True,
            'clip_model_path': config.get('clip_model_path'),
            'fusion_method': 'weighted',
            'position_encoding': 'learned',
            'text_weight': 0.6,  # BGE权重
            'visual_weight': 0.4,  # CLIP权重
            'device': 'cuda',
        }

        multimodal_retriever = SelfAwareMultimodalRetriever(
            config=multimodal_config,
            text_retriever=bge_retriever,
            visual_retriever=clip_retriever
        )
        print("   ✓ 多模态融合检索器创建成功")

        print("\n✅ 多模态检索器初始化成功！")
        return True

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multimodal_retriever()
    if success:
        print("\n✅ 所有测试通过")
    else:
        print("\n✗ 测试失败")