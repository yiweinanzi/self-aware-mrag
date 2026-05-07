#!/bin/bash
#SBATCH --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=32G -p 5090
#SBATCH --job-name=build_clip_index
#SBATCH --output=build_clip_index_%j.out
#SBATCH --error=build_clip_index_%j.err
#SBATCH --time=02:00:00

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "构建MultiModalQA CLIP索引"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 只运行CLIP索引构建部分
python -c "
import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np

print('加载已保存的文档映射...')
with open('/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/documents.json', 'r') as f:
    documents = []
    for line in f:
        if line.strip():
            documents.append(json.loads(line))

print(f'总共加载 {len(documents)} 个文档')

# 加载CLIP模型
print('\n1. 加载CLIP模型...')
model_path = '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336'
model = CLIPModel.from_pretrained(model_path).cuda()
processor = CLIPProcessor.from_pretrained(model_path)
print(f'   模型: {model_path}')

# 准备文本
print('\n2. 准备文本...')
texts = []
valid_doc_ids = []

for i, doc in enumerate(documents):
    # 文本：标题 + 内容的精简版本
    text = f\"Document: {doc['title']}\"
    if 'text' in doc:
        text = f\"{text}\\n{doc['text'][:200]}...\"

    texts.append(text)
    valid_doc_ids.append(doc['docid'])

    if (i + 1) % 10000 == 0:
        print(f'   已准备 {i+1:,} 个文本')

print(f'   文本数: {len(texts)}')

# 使用CLIP编码
print('\n3. 使用CLIP编码...')
batch_size = 64
all_text_features = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]

    with torch.no_grad():
        # 编码文本
        text_inputs = processor(
            text=batch_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(model.device)

        text_features = model.get_text_features(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_text_features.append(text_features.cpu().numpy())

    print(f'   处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}')

# 合并所有特征
all_text_features = np.vstack(all_text_features)
print(f'✅ 文本特征维度: {all_text_features.shape}')

# 构建FAISS索引
print('\n4. 构建FAISS索引...')
dimension = all_text_features.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(all_text_features)

# 保存索引
output_dir = '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa'
index_path = os.path.join(output_dir, 'clip_Flat.index')
faiss.write_index(index, index_path)

# 保存映射信息
mapping = {
    'doc_ids': valid_doc_ids,
    'texts': texts[:len(valid_doc_ids)]
}

with open(os.path.join(output_dir, 'clip_mapping.json'), 'w') as f:
    json.dump(mapping, f, indent=2)

print(f'\n✅ CLIP索引构建完成:')
print(f'   - 索引文件: {index_path}')
print(f'   - 文本数: {len(texts)}')
print(f'   - 维度: {dimension}')
"

echo -e "\nCLIP索引构建完成时间: $(date)"