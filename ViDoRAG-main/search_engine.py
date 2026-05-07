from typing import Optional, List, Mapping, Any, Dict
import json
from tqdm import tqdm
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.mixture import GaussianMixture

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import NodeWithScore, BaseNode, MetadataMode, IndexNode, ImageNode, TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llms.vl_embedding import VL_Embedding
from utils.format_converter import nodefile2node,nodes2dict


def gmm(recall_result: list[NodeWithScore], input_length: int=20, max_valid_length: int=10, min_valid_length: int=5) -> List[NodeWithScore]:
    
    scores = [node.score for node in recall_result[:input_length]]
    
    scores = np.array(scores)
    scores = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=1,random_state=0)
    gmm.fit(scores)
    labels = gmm.predict(scores)

    scores = scores.flatten()
    scores = [scores[labels == label] for label in np.unique(labels)]
    recall_result = [np.array(recall_result[:input_length])[labels == label].tolist() for label in np.unique(labels)]

    max_values = np.array([np.max(p) for p in scores])
    sorted_indices = np.argsort(-max_values)
    
    if len(sorted_indices) == 1:
        valid_recall_result = recall_result[0]
        valid_recall_result = valid_recall_result[:max_valid_length]
        for node in valid_recall_result:
            node.score = None
        return valid_recall_result
    
    max_index = sorted_indices[0]
    second_max_index = sorted_indices[1]
    
    valid_recall_result = recall_result[max_index]

    if len(valid_recall_result) > max_valid_length:
        valid_recall_result = valid_recall_result[:max_valid_length]
    elif len(valid_recall_result) < min_valid_length:
        second_valid_recall_result_len = min_valid_length - len(valid_recall_result)
        valid_recall_result.extend(recall_result[second_max_index][:second_valid_recall_result_len])    
    
    for node in valid_recall_result:
        node.score = None

    return valid_recall_result

class SearchEngine:
    def __init__(self,dataset, node_dir_prefix=None,embed_model_name='BAAI/bge-m3'):# nvidia/NV-Embed-v2 "vidore/colqwen2-v0.1"
        Settings.llm = None
        self.gmm=False
        self.gmm_candidate_length = False
        self.return_raw = False
        self.input_gmm = 20
        self.max_output_gmm = 10
        self.min_output_gmm = 5
        self.dataset = dataset
        self.dataset_dir = os.path.join('./data', dataset)
        if node_dir_prefix is None:
            if 'bge' in embed_model_name:
                node_dir_prefix = 'bge_ingestion'
            elif 'NV-Embed' in embed_model_name:
                node_dir_prefix = 'nv_ingestion'
            elif 'colqwen' in embed_model_name:
                node_dir_prefix = 'colqwen_ingestion'
            elif 'openbmb' in embed_model_name:
                node_dir_prefix = 'visrag_ingestion'
            elif 'colpali' in embed_model_name:
                node_dir_prefix = 'colpali_ingestion'
            else:
                raise ValueError('Please specify the node_dir_prefix')
        
        if node_dir_prefix in ['colqwen_ingestion','visrag_ingestion','colpali_ingestion']:
            self.vl_ret = True
        else:
            self.vl_ret = False
        
        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.rag_dataset_path = os.path.join(self.dataset_dir, 'rag_dataset.json')
        self.workers = 1
        self.embed_model_name = embed_model_name
        if 'vidore' in embed_model_name or 'openbmb' in embed_model_name:
            if self.vl_ret:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
            else:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='text')
        else:
            self.vector_embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, embed_batch_size=10, max_length=512, trust_remote_code=True, device='cuda')
        self.recall_num = 100
        self.query_engine = self.load_query_engine()
        self.output_dir = os.path.join(self.dataset_dir, 'search_output')
        # os.makedirs(self.output_dir, exist_ok=True)
    
    def online_search(self,query,node_list,topk=9):
        nodes = [TextNode.from_dict(node['node']) for node in node_list]
        vector_index = VectorStoreIndex(nodes, embed_model= self.vector_embed_model, show_progress=True, use_async=False, insert_batch_size=2048)
        vector_retriever = vector_index.as_retriever(similarity_top_k=topk)
        node_postprocessors = self.load_node_postprocessors()
        query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            node_postprocessors=node_postprocessors
        )
        query_bundle = QueryBundle(query_str=query)
        recall_results = query_engine.retrieve(query_bundle)
        return nodes2dict(recall_results)
        
    def load_nodes(self):
        files = os.listdir(self.node_dir)
        parsed_files = []
        max_workers = 10
        if max_workers == 1:
            for file in tqdm(files):
                input_file = os.path.join(self.node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    continue
                nodes = nodefile2node(input_file)
                parsed_files.extend(nodes)
        else:
            def parse_file(file,node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    return []
                return nodefile2node(input_file)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # results = list(tqdm(executor.map(parse_file, files, self.node_dir), total=len(files)))
                results = list(tqdm(executor.map(parse_file, files, [self.node_dir]*len(files)), total=len(files)))
            for result in results:
                parsed_files.extend(result)
        return parsed_files
        
    def load_query_engine(self):
        print('Loading nodes...')
        self.nodes = self.load_nodes()
        if self.vl_ret and 'vidore' in self.embed_model_name:
            self.embedding_img = [torch.tensor(node.embedding).view(-1,128).bfloat16() for node in self.nodes]
            self.embedding_img = [tensor.to(self.vector_embed_model.embed_model.device) for tensor in self.embedding_img]
        else:
            retriever = self.load_retriever_embed(self.nodes)
            node_postprocessors = self.load_node_postprocessors()
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=node_postprocessors
            )
            return query_engine
        
    def load_node_postprocessors(self):
        return []
    
    def load_retriever_embed(self, nodes):
        vector_index = VectorStoreIndex(nodes, embed_model= self.vector_embed_model, show_progress=True, use_async=False, insert_batch_size=2048)
        vector_retriever = vector_index.as_retriever(similarity_top_k=self.recall_num)
        return vector_retriever
    
    def search(self, query):
        if self.vl_ret and 'vidore' in self.embed_model_name:
            query_embedding = self.vector_embed_model.embed_text(query)
            scores = self.vector_embed_model.processor.score(query_embedding,self.embedding_img)
            k = min(100, scores[0].numel())
            values, indices = torch.topk(scores[0], k=k)
            recall_results = [self.nodes[i] for i in indices]
            for node in recall_results:
                node.embedding = None
            recall_results = [NodeWithScore(node=node, score=score) for node, score in zip(recall_results, values)]
            recall_results_output = recall_results
        else:
            query_bundle = QueryBundle(query_str=query)
            recall_results = self.query_engine.retrieve(query_bundle)
            recall_results_output = recall_results
        if self.gmm:
            recall_results_output = gmm(recall_results,self.input_gmm,self.max_output_gmm,self.min_output_gmm)
        if self.return_raw:
            return recall_results_output
        if self.gmm_candidate_length:
            candidate_length = [1,2,4,6,9,12,16,20]
            current_length = len(recall_results_output)
            target_length = min([num for num in candidate_length if num > current_length])
            recall_results_output = recall_results[:target_length]
        return nodes2dict(recall_results_output)
    
    def search_example(self,example):
        query = example['query']
        recall_result =  self.search(query)
        example['recall_result'] = recall_result
        return example
    
    def search_multi_session(self,output_file='search_result.json'):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.rag_dataset_path, 'r') as f:
            dataset = json.load(f)
        data = dataset['examples']
        results = []
        if self.workers == 1:
            for example in tqdm(data):
                results.append(self.search_example(example))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.search_example, example): example for example in data}
                for future in tqdm(as_completed(future_to_file), total=len(data), desc='Processing files'):
                    results.append(future.result())
        with open(os.path.join(self.output_dir, output_file), 'w') as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)

class HybridSearchEngine:
    
    def __init__(self,
                 dataset,
                 node_dir_prefix_vl = None,
                 node_dir_prefix_text = None,
                 embed_model_name_vl = 'vidore/colqwen2-v1.0',
                 embed_model_name_text = 'BAAI/bge-m3',
                 topk=10,
                 gmm=False):
        self.dataset = dataset
        self.dataset_dir = os.path.join('./data', dataset)
        self.img_dir = os.path.join(self.dataset_dir, 'img')
        self.ppocr_dir = os.path.join(self.dataset_dir, 'bge_ingestion')
        self.engine_vl = SearchEngine(dataset,node_dir_prefix=node_dir_prefix_vl,embed_model_name=embed_model_name_vl)
        self.engine_text = SearchEngine(dataset,node_dir_prefix=node_dir_prefix_text,embed_model_name=embed_model_name_text)
        self.topk = topk
        self.gmm = gmm
    
    def search(self,query):
        union_result = False
        if union_result:
            if self.gmm:
                self.engine_vl.gmm = True
                self.engine_text.gmm = True
                self.engine_vl.input_gmm = self.topk *2
                self.engine_text.input_gmm = self.topk *2
                self.engine_vl.max_output_gmm = self.topk
                self.engine_text.max_output_gmm = self.topk
                self.engine_vl.min_output_gmm = self.topk//2
                self.engine_text.min_output_gmm = self.topk//2
            result_vl = self.engine_vl.search(query)
            result_text = self.engine_text.search(query)
            result_vl['source_nodes'] = result_vl['source_nodes'][:self.topk]
            result_text['source_nodes'] = result_text['source_nodes'][:self.topk]

            result_docs = dict()
            for node in result_vl['source_nodes']:
                file = os.path.basename(node['node']['image_path']).split('.')[0]
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))
                
            for node in result_text['source_nodes']:
                file = node['node']['metadata']['filename'].split('.')[0]
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))
            
            recall_result = []
            
            for key,pages in result_docs.items():
                # from small to big
                pages = sorted(pages)
                for page in pages:
                    with open(os.path.join(self.ppocr_dir,f'{key}_{page}.node'),'r') as f:
                        text = json.load(f)
                    text = ' '.join([item['text'] for item in text])
                        
                    file_path = os.path.join(self.img_dir,f'{key}_{page}.jpg')
                    
                    node = ImageNode(image_path=file_path,text=text,metadata=dict(file_name=file_path))
                    
                    recall_result.append(NodeWithScore(node=node,score=None))
            
            return nodes2dict(recall_result)
        
        else:
            
            self.engine_vl.return_raw = True
            self.engine_text.return_raw = True
            
            result_vl = self.engine_vl.search(query)
            result_text = self.engine_text.search(query)
            
            result_vl_gmm = gmm(result_vl,self.topk * 2,self.topk, 5)
            result_text_gmm = gmm(result_text,self.topk * 2,self.topk, 5)

            result_vl_gmm = nodes2dict(result_vl_gmm)
            result_text_gmm = nodes2dict(result_text_gmm)

            
            result_docs = dict()
            for node in result_vl_gmm['source_nodes']:
                # file = os.path.basename(node['node']['image_path']).split('.')[0]
                file = '.'.join(os.path.basename(node['node']['image_path']).split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))
            for node in result_text_gmm['source_nodes']:
                # file = node['node']['metadata']['filename'].split('.')[0]
                file = '.'.join(node['node']['metadata']['filename'].split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))
            
            result_docs_list = []
            for key,pages in result_docs.items():
                for page in pages:
                    result_docs_list.append(f'{key}_{page}')
            
            result_vl = nodes2dict(result_vl)
            result_text = nodes2dict(result_text)
            result_docs_text_list = []
            result_docs_vl_list = []
            for node in result_vl['source_nodes']:
                # file = os.path.basename(node['node']['image_path']).split('.')[0]
                file = '.'.join(os.path.basename(node['node']['image_path']).split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                result_docs_vl_list.append(doc+'_'+page)
            for node in result_text['source_nodes']:
                # file = node['node']['metadata']['filename'].split('.')[0]
                file = '.'.join(node['node']['metadata']['filename'].split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                result_docs_text_list.append(doc+'_'+page)

            overleap = [doc for doc in result_docs_vl_list if doc in result_docs_text_list]
            
            candidate_length = [1,2,4,6,9,12,16,20]
            already_length = sum([len(value) for _,value in result_docs.items()])
            target_length = min((x for x in candidate_length if x >= already_length))
            candidate_overleap = [node for node in overleap if node not in result_docs_list][:target_length-already_length]
            candidate_overleap = candidate_overleap + [node for node in result_docs_vl_list if (node not in candidate_overleap) and (node not in result_docs_list)]
            candidate_overleap = candidate_overleap[:target_length-already_length]

            for file in candidate_overleap:
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))

            recall_result = []
            for key,pages in result_docs.items():
                pages = sorted(pages)
                for page in pages:
                    with open(os.path.join(self.ppocr_dir,f'{key}_{page}.node'),'r') as f:
                        text = json.load(f)
                    text = ' '.join([item['text'] for item in text])
                    file_path = os.path.join(self.img_dir,f'{key}_{page}.jpg')
                    node = ImageNode(image_path=file_path,text=text,metadata=dict(file_name=file_path))
                    recall_result.append(NodeWithScore(node=node,score=None))
            return nodes2dict(recall_result)
            
    
    

if __name__ == '__main__':
    datasets = ['ExampleDataset']
    for dataset in datasets:
        # search_engine = SearchEngine(dataset,node_dir_prefix='visrag_ingestion',embed_model_name='openbmb/VisRAG-Ret')
        # search_engine = SearchEngine(dataset,node_dir_prefix='nv_ingestion',embed_model_name='nvidia/NV-Embed-v2')
        search_engine = HybridSearchEngine(dataset,gmm=True)
        search_engine.search('ok')
        import pdb;pdb.set_trace()
    

    