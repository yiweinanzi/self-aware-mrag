# Retrieval Setup for Five Datasets

本文档详细说明五个数据集的检索配置设置，确保baseline对比的公平性。

---

## 1. 数据集概览

| 数据集 | 任务类型 | 检索语料库 | 图像来源 | 测试集规模 |
|--------|----------|------------|----------|------------|
| **OK-VQA** | 开放域视觉问答 | Wikipedia (3M) | Wikipedia内嵌图像 | 5,040 |
| **A-OKVQA** | 开放域视觉问答 | Wikipedia (3M) | COCO图像 | 6,702 |
| **MRAG-Bench** | 多模态RAG评估 | 自带图像库 | 自带高质图像 | 600 |
| **MultiModalQA** | 给定文档多模态QA | 数据集自带 | 数据集自带 | ~10K |
| **WebQA** | Web视觉问答 | 数据集自带事实 | 数据集自带图像 | 7,540 |

---

## 2. 语料库详情

### 2.1 Wikipedia 3M Corpus (用于 OK-VQA, A-OKVQA)

| 项目 | 详情 |
|------|------|
| **语料库路径** | `/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl` |
| **文档数量** | ~3,000,000 |
| **文本索引** | `/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index` |
| **索引大小** | 12 GB |
| **索引类型** | FAISS Flat (精确搜索) |
| **嵌入模型** | BGE-large-en-v1.5 (1024维) |
| **CLIP图像索引** | `/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index` |
| **Chunk策略** | 按段落切分，每段不超过512 tokens |

### 2.2 A-OKVQA 数据集

| 项目 | 详情 |
|------|------|
| **数据集路径** | `/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA_COPY/aokvqa` |
| **训练集** | 17,056 samples |
| **验证集** | 1,145 samples |
| **测试集** | 6,702 samples |
| **图像来源** | COCO 2017 |
| **COCO目录** | `/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA_COPY/coco` |
| **问题类型** | 多选题 (4选项) |
| **检索语料库** | Wikipedia 3M (与OK-VQA共用) |

### 2.3 MRAG-Bench Corpus

| 项目 | 详情 |
|------|------|
| **数据集路径** | `/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw` |
| **测试集大小** | 600 samples |
| **图像库路径** | `/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/image_corpus.jsonl` |
| **图像数量** | 19,178 |
| **CLIP索引** | `/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/clip_image_Flat.index` |
| **索引大小** | 57 MB |
| **图像元数据** | `/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/image_metadata.json` |
| **图像格式** | 高质量场景图，自带详细caption |

### 2.4 MultiModalQA Corpus

| 项目 | 详情 |
|------|------|
| **数据集路径** | `/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA` |
| **文本/表格语料库** | `/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/corpus.jsonl` |
| **文档数量** | 228,327 |
| **图像语料库** | `/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/image_corpus.jsonl` |
| **图像数量** | 57,058 |
| **BGE文本索引** | `/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index` (892 MB) |
| **CLIP图像索引** | `/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/clip_Flat.index` (669 MB) |
| **文档映射** | `/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/doc_mapping.json` |
| **检索方式** | 使用数据集自带文档ID（非开放域检索） |

### 2.5 WebQA 数据集

| 项目 | 详情 |
|------|------|
| **数据集路径** | `/data0/home/zqwang/ACL/FlashRAG/flashrag/data/WebQA/data_unzip` |
| **测试集文件** | `WebQA_test.json` |
| **训练集文件** | `WebQA_train_val.json` |
| **测试集条目数** | 7,540 |
| **训练集条目数** | 41,732 |
| **数据结构** | 字典格式，以GUID为键 |
| **文本事实总数** | ~120,699 (测试集) |
| **图像事实总数** | ~107,323 (测试集) |
| **检索方式** | 使用数据集自带的txt_Facts和img_Facts（非开放域检索） |

**WebQA数据结构示例：**
```json
{
  "guid": {
    "Q": "问题文本",
    "A": ["答案"],
    "split": "test",
    "Guid": "guid",
    "txt_Facts": [
      {
        "title": "文档标题",
        "fact": "事实内容",
        "url": "来源URL"
      }
    ],
    "img_Facts": [
      {
        "image_id": 12345,
        "title": "图像标题",
        "caption": "图像描述",
        "url": "图像URL"
      }
    ]
  }
}
```

---

## 3. 检索参数配置

### 3.1 通用检索参数 (所有数据集)

| 参数 | 值 | 说明 |
|------|-----|------|
| **retrieval_topk** | 5 | 检索返回的文档数量 |
| **retrieval_model** | BGE-large-en-v1.5 | 文本嵌入模型 |
| **embedding_dim** | 1024 | BGE嵌入维度 |
| **max_query_length** | 512 | 查询最大长度 |
| **pooling_method** | cls/mean | BGE池化方式 |
| **faiss_index_type** | Flat | 精确搜索（非IVF） |

### 3.2 多模态检索参数

| 参数 | OK-VQA/A-OKVQA | MRAG-Bench | MultiModalQA | WebQA |
|------|----------------|------------|---------------|-------|
| **use_multimodal_retrieval** | True | True | True | False |
| **text_weight (BGE)** | 0.6 | - | 0.6 | - |
| **visual_weight (CLIP)** | 0.4 | 1.0 | 0.4 | - |
| **clip_model** | CLIP-ViT-L-14-336 | CLIP-ViT-L-14-336 | CLIP-ViT-L-14-336 | - |
| **clip_index_path** | wiki_3m/clip | mrag_bench/clip | multimodalqa/clip | - |

### 3.3 Reranker配置 (部分实验)

| 参数 | 值 |
|------|-----|
| **use_reranker** | True (部分实验) |
| **reranker_model** | bge-reranker-v2-m3 |
| **rerank_topk** | 5 |
| **rerank_max_length** | 512 |

---

## 4. Chunk切分策略

### 4.1 Wikipedia 3M Corpus

| 项目 | 策略 |
|------|------|
| **切分方式** | 按段落 (paragraph) |
| **最大长度** | 512 tokens |
| **重叠** | 无重叠 |
| **文档粒度** | 单个段落/节 |
| **保留格式** | 保留标题层级 |

### 4.2 MRAG-Bench

| 项目 | 策略 |
|------|------|
| **切分方式** | 单图即文档 |
| **文档粒度** | 每张图+caption为一个文档 |
| **图像格式** | 原始分辨率 + 预生成的caption |

### 4.3 MultiModalQA

| 项目 | 策略 |
|------|------|
| **文本文档** | Wikipedia段落级别 |
| **表格文档** | 整表作为一个文档 |
| **表格处理** | 简化为文本序列 (MOQAGPT风格) |
| **图像文档** | 单图即文档 |

### 4.4 WebQA

| 项目 | 策略 |
|------|------|
| **切分方式** | 不需要切分（使用自带事实） |
| **文档粒度** | 每个fact作为一个文档 |
| **文本事实** | 直接使用txt_Facts |
| **图像事实** | 直接使用img_Facts |

---

## 5. Baseline公平性保证

### 5.1 统一检索器

所有baseline方法使用**相同的检索器配置**：

| 方法 | 检索器 | 候选池 | top-k |
|------|--------|--------|-------|
| Self-Aware-MRAG | BGE+CLIP Fusion | 同数据集语料库 | 5 |
| SAM-RAG | BGE+CLIP Fusion | 同数据集语料库 | 5 |
| mR²AG | BGE+CLIP Fusion | 同数据集语料库 | 5 |
| VisRAG | BGE+CLIP Fusion | 同数据集语料库 | 5 |
| ViDoRAG | BGE+CLIP Fusion | 同数据集语料库 | 5 |
| RagVL | BGE+CLIP Fusion | 同数据集语料库 | 5 |
| MuRAG | BGE+CLIP Fusion | 同数据集语料库 | 5 |

### 5.2 统一预算

| 资源 | 限制 |
|------|------|
| **检索文档数** | k=5 (所有方法一致) |
| **输入图像** | 最多20张 |
| **最大tokens** | 50 (OK-VQA), 50 (MultiModalQA) |
| **GPU数量** | 2x 5090 |

---

## 6. 各数据集特有配置

### 6.1 OK-VQA / A-OKVQA

```python
'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index'
'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl'
'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index'
'retrieval_topk': 5
'use_multimodal_retrieval': True
'text_retrieval_weight': 0.6
'visual_retrieval_weight': 0.4
```

**A-OKVQA额外配置：**
```python
'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA_COPY/aokvqa'
'coco_image_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA_COPY/coco'
```

### 6.2 MRAG-Bench

```python
'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw'
'image_corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/image_corpus.jsonl'
'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/clip_image_Flat.index'
'retrieval_topk': 5
'use_multimodal_retrieval': True
```

### 6.3 MultiModalQA

```python
'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA'
'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index'
'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/corpus.jsonl'
'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa'
'image_corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/image_corpus.jsonl'
'use_dataset_docs': True  # 使用数据集提供的文档，非开放域检索
'simple_table_processing': True  # 表格简化为文本
```

### 6.4 WebQA

```python
'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/WebQA/data_unzip'
'test_file': 'WebQA_test.json'
'train_file': 'WebQA_train_val.json'
'use_dataset_facts': True  # 使用数据集自带的txt_Facts和img_Facts
'no_external_retrieval': True  # 不使用外部检索
```

---

## 7. 索引构建命令

### Wikipedia 3M BGE Index
```bash
# 已构建，位于: /data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index
# 大小: 12 GB
# 文档数: ~3,000,000
```

### MRAG-Bench CLIP Index
```bash
python build_mrag_image_indexes.py
# 输出: /data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/clip_image_Flat.index
# 大小: 57 MB
# 图像数: 19,178
```

### MultiModalQA Indexes
```bash
python build_multimodalqa_indexes_complete.py
# 输出:
#   - bge_Flat.index (892 MB)
#   - clip_Flat.index (669 MB)
#   - corpus.jsonl (228K docs)
#   - image_corpus.jsonl (57K images)
```

### WebQA 数据解压
```bash
cd /data0/home/zqwang/ACL/FlashRAG/flashrag/data/WebQA
7z x WebQA_data_first_release.7z -odata_unzip
# 输出:
#   - WebQA_test.json (7,540 条目)
#   - WebQA_train_val.json (41,732 条目)
```

---

## 8. 注意事项

1. **MultiModalQA是给定文档任务**，不使用Wikipedia开放域检索，而是使用数据集自带的文档ID映射

2. **WebQA是给定事实任务**，不使用外部检索，而是使用数据集自带的txt_Facts和img_Facts

3. **MRAG-Bench自带高质量图像库**，图像已预生成详细caption，检索质量高于通用Wikipedia图像

4. **A-OKVQA使用COCO图像**，需要额外下载COCO 2017图像数据

5. **所有baseline使用相同检索器和候选池**，确保公平对比

6. **多模态融合权重固定为 BGE 60% + CLIP 40%**，跨所有数据集保持一致

7. **Reranker仅在部分实验中使用**，使用时需在论文中明确说明

---

## 9. 数据集规模汇总

| 数据集 | Train | Val | Test | 总计 |
|--------|-------|-----|------|------|
| OK-VQA | - | - | 5,040 | 9,057 |
| A-OKVQA | 17,056 | 1,145 | 6,702 | 24,903 |
| MRAG-Bench | - | - | 600 | 600 |
| MultiModalQA | - | - | ~10K | ~40K |
| WebQA | 41,732 | - | 7,540 | 49,272 |

---

*生成时间: 2025-12-30*
*配置版本: v1.1*
