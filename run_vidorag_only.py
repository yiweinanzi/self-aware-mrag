#!/usr/bin/env python3
"""
Run ViDoRAG only with debug
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONPATH'] = '/data0/home/zqwang/ACL:/data0/home/zqwang/ACL/FlashRAG'

# Import the run script with modifications
import experiments.run_okvqa_baselines as baselines_module
run_baseline_experiment = baselines_module.run_baseline_experiment

# Config for testing ViDoRAG only
config = {
    'dataset': 'okvqa',
    'max_samples': 3,  # Only 3 samples
    'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'max_new_tokens': 20,
    'retrieval_topk': 5,
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'use_multimodal_retrieval': True,
    'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
    'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index',
    'text_retrieval_weight': 0.6,
    'visual_retrieval_weight': 0.4,
    'uncertainty_threshold': 0.43,
    'text_weight': 0.4,
    'visual_weight': 0.3,
    'alignment_weight': 0.3,
    'use_improved_estimator': True,
    'output_dir': 'results_vidorag_debug'
}

# Only run ViDoRAG
methods = ['ViDoRAG']

print("Running ViDoRAG with debug...")
results = run_baseline_experiment(config=config, methods=methods)

print("\nResults:")
for method_name, result in results.items():
    print(f"{method_name}: accuracy={result.get('accuracy', 0):.2%}")