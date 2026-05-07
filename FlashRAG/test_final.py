import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 导入主程序
from experiments.run_okvqa_baselines import main

# 设置参数
sys.argv = [
    'run_okvqa_baselines.py',
    '--dataset', 'okvqa',
    '--max-samples', '3',
    '--model-path', '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    '--torch-dtype', 'bfloat16',
    '--max-new-tokens', '20',
    '--retrieval-topk', '5',
    '--faiss-index-path', '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    '--corpus-path', '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    '--retrieval-model-path', '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    '--use-multimodal-retrieval',
    '--clip-model-path', '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
    '--clip-index-path', '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index',
    '--text-retrieval-weight', '0.6',
    '--visual-retrieval-weight', '0.4',
    '--uncertainty-threshold', '0.43',
    '--text-weight', '0.4',
    '--visual-weight', '0.3',
    '--alignment-weight', '0.3',
    '--use-improved-estimator',
    '--output-dir', 'results_final_test',
    '--save-detailed-results',
    '--save-sample-results',
    '--enable-complete-metrics',
    '--methods', 'MuRAG', 'VisRAG'
]

# 运行
main()
