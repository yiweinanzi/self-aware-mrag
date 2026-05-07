import os
import json
from tqdm import tqdm
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core.schema import NodeWithScore, ImageNode

from search_engine import SearchEngine,HybridSearchEngine

from llms.llm import LLM
from llms.evaluator import Evaluator

from utils.overall_evaluator import eval_search,eval_search_type_wise
from vidorag_agents import ViDoRAG_Agents

class MMRAG:
    def __init__(self,
                dataset='ExampleDataset',
                query_file='rag_dataset.json',
                experiment_type = 'retrieval_infer',
                generate_vlm='qwen-vl-max',
                embed_model_name='BAAI/bge-m3',
                embed_model_name_vl=None, # openbmb/VisRAG-Ret vidore/colqwen2-v1.0
                embed_model_name_text=None, # nvidia/NV-Embed-v2 BAAI/bge-m3
                workers_num = 1,
                topk=10):
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.dataset = dataset
        self.query_file = query_file
        self.dataset_dir = os.path.join('./data', dataset)
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(self.dataset_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.vlm = LLM(model_name=generate_vlm)
        self.evaluator = Evaluator()
        
        # load search_engine
        if embed_model_name_vl is not None and embed_model_name_text is not None:
            self.search_engine = HybridSearchEngine(self.dataset,
                                                    embed_model_name_vl=embed_model_name_vl,
                                                    embed_model_name_text=embed_model_name_text,
                                                    topk=topk)
        else:
            self.search_engine = SearchEngine(self.dataset, embed_model_name=embed_model_name)

        # retrieval only
        if experiment_type == 'retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'base_retrieval_{embed_model_name}.jsonl'
        # hybrid retrieval
        elif experiment_type == 'dynamic_hybird_retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.search_engine.gmm = True
            self.output_file_name = f'dynamic_hybird_retrieval_{embed_model_name_vl}_{embed_model_name_text}.jsonl'
        # vidorag
        elif experiment_type == 'vidorag':
            self.agents = ViDoRAG_Agents(self.vlm)
            self.eval_func = self.vidorag
            self.search_engine.gmm = True
            self.search_engine.gmm_candidate_length = True
            self.output_file_name = f'vidorag_{generate_vlm}.jsonl'
        
        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/","-"))

    def retrieval_infer(self,sample):
        query = sample['query']
        recall_results = self.search_engine.search(query)
        recall_results['source_nodes'] = recall_results['source_nodes'][:100]
        sample['recall_results'] = recall_results
        return sample
    def vidorag(self,sample):
        query = sample['query']
        print(query)
        recall_results = self.search_engine.search(query)
        candidate_image = [os.path.join(self.img_dir, os.path.basename(node['node']['metadata'].get('file_name',node['node']['metadata'].get('filename')).replace('.txt','.jpg'))) for node in recall_results['source_nodes']]
        if 'gmm' not in self.experiment_type:
            candidate_image = candidate_image[:self.top_k]
        try:
            answer = self.agents.run_agent(query, candidate_image)
        except Exception as e:
            print(e)
            return None
        
        sample['eval_result'] = self.evaluator.evaluate(query, sample['reference_answer'], str(answer))
        sample['response'] = answer
        sample['recall_results'] = dict(
            source_nodes=[NodeWithScore(node=ImageNode(image_path=image,metadata=dict(file_name=image)), score=None).to_dict() for image in candidate_image],
            response=None,
            metadata=None)
        return sample


    def eval_dataset(self):
        eval_func = self.eval_func
        
        rag_dataset_path = os.path.join(self.dataset_dir,self.query_file)
        with open(rag_dataset_path, "r") as f:
            data = json.load(f)
        data = data['examples']
        
        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
            uid_already = [item['uid'] for item in results]
            data = [item for item in data if item['uid'] not in uid_already]
            
        if self.workers_num == 1:
            for item in tqdm(data):
                result = eval_func(item)
                if result is None:
                    continue
                with open(self.output_file_path, "a") as f:
                    json.dump(result, f,ensure_ascii=False)
                    f.write("\n")
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    result = future.result()
                    results.append(result)
                    if len(results) >= 3:
                        with open(self.output_file_path, "a") as f:
                            for res in results:
                                if res is None:
                                    continue
                                f.write(json.dumps(res,ensure_ascii=False) + "\n")
                        results = []
                if results:
                    with open(self.output_file_path, "a") as f:
                        for res in results:
                            if res is None:
                                continue
                            f.write(json.dumps(res,ensure_ascii=False) + "\n")

        return self.output_file_path

    def eval_overall(self):
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search(data)
        with open(self.output_file_path.replace(".jsonl", "_eval.json"), "w") as f:
            json.dump(results, f,indent=2,ensure_ascii=False)
    
    def eval_overall_type_wise(self):
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search_type_wise(data)
        with open(self.output_file_path.replace(".jsonl", "_eval_type_wise.json"), "w") as f:
            json.dump(results, f,indent=2,ensure_ascii=False)
        
def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ExampleDataset', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='rag_dataset.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='retrieval_infer', help="The type of experiment")
    parser.add_argument("--embed_model_name", type=str, default='BAAI/bge-m3', help="The name of embedding model")
    parser.add_argument("--workers_num", type=int, default=1, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    parser.add_argument("--embed_model_name_vl", type=str, default=None, help="The name of embedding model for vl")
    parser.add_argument("--embed_model_name_text", type=str, default=None, help="The name of embedding model for text")
    parser.add_argument("--generate_vlm", type=str, default='qwen-vl-max', help="The name of VLM model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    mmrag = MMRAG(
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        embed_model_name=args.embed_model_name,
        workers_num=args.workers_num,
        topk=args.topk,
        embed_model_name_vl=args.embed_model_name_vl,
        embed_model_name_text=args.embed_model_name_text,
        generate_vlm=args.generate_vlm
    )
    mmrag.eval_dataset()
    mmrag.eval_overall()
    mmrag.eval_overall_type_wise()