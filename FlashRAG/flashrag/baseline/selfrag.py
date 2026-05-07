"""
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-reflection (ICLR 2024)
Paper: https://arxiv.org/abs/2310.11511
Code: https://github.com/AkariAsai/self-rag

This implementation adapts Self-RAG for multimodal VQA tasks.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import warnings

# Control tokens used by Self-RAG
CONTROL_TOKENS = [
    "[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"
]


class SelfRAG:
    """
    Self-RAG implementation for VQA tasks.
    
    Unlike standard RAG, Self-RAG:
    1. Decides whether to retrieve on-demand
    2. Uses reflection tokens to critique its own generation
    3. Scores outputs based on relevance, groundedness, and utility
    """
    
    def __init__(
        self,
        model_name: str = "selfrag/selfrag_llama2_7b",
        threshold: float = 0.2,
        w_rel: float = 1.0,
        w_sup: float = 1.0,
        w_use: float = 0.5,
        max_new_tokens: int = 100,
        use_groundness: bool = True,
        use_utility: bool = True,
        use_seqscore: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize Self-RAG model.
        
        Args:
            model_name: HuggingFace model name
            threshold: Threshold for adaptive retrieval (0.0-1.0)
            w_rel: Weight for relevance score
            w_sup: Weight for support/groundedness score
            w_use: Weight for utility score
            max_new_tokens: Maximum number of tokens to generate
            use_groundness: Whether to use groundedness critique
            use_utility: Whether to use utility critique
            use_seqscore: Whether to use sequence score
            device: Device to run on
        """
        self.model_name = model_name
        self.threshold = threshold
        self.w_rel = w_rel
        self.w_sup = w_sup
        self.w_use = w_use
        self.max_new_tokens = max_new_tokens
        self.use_groundness = use_groundness
        self.use_utility = use_utility
        self.use_seqscore = use_seqscore
        self.device = device
        
        # Load model using vllm for efficiency
        try:
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.model = None  # Lazy initialization
        except ImportError:
            warnings.warn("vllm not installed. Self-RAG requires vllm for efficient inference.")
            self.LLM = None
            self.SamplingParams = None
            self.model = None
        
        # Special tokens
        self.ret_tokens = None
        self.rel_tokens = None
        self.grd_tokens = None
        self.ut_tokens = None
    
    def _initialize_model(self):
        """Lazy initialization of the model."""
        if self.model is None and self.LLM is not None:
            print(f"Loading Self-RAG model: {self.model_name}")
            self.model = self.LLM(self.model_name, dtype="half", gpu_memory_utilization=0.8)
            self._load_special_tokens()
            print("Self-RAG model loaded successfully!")
    
    def _load_special_tokens(self):
        """Load special token IDs for reflection tokens."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Retrieval tokens
        self.ret_tokens = {
            "[Retrieval]": tokenizer.convert_tokens_to_ids("[Retrieval]"),
            "[No Retrieval]": tokenizer.convert_tokens_to_ids("[No Retrieval]")
        }
        
        # Relevance tokens
        self.rel_tokens = {
            "[Relevant]": tokenizer.convert_tokens_to_ids("[Relevant]"),
            "[Irrelevant]": tokenizer.convert_tokens_to_ids("[Irrelevant]")
        }
        
        # Groundedness tokens
        self.grd_tokens = {
            "[Fully supported]": tokenizer.convert_tokens_to_ids("[Fully supported]"),
            "[Partially supported]": tokenizer.convert_tokens_to_ids("[Partially supported]"),
            "[No support / Contradictory]": tokenizer.convert_tokens_to_ids("[No support / Contradictory]")
        }
        
        # Utility tokens
        self.ut_tokens = {
            f"[Utility:{i}]": tokenizer.convert_tokens_to_ids(f"[Utility:{i}]")
            for i in range(1, 6)
        }
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Format prompt for Self-RAG.
        
        Args:
            question: The input question
            context: Optional retrieved context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        if context is not None:
            prompt += f"[Retrieval]<paragraph>{context}</paragraph>"
        return prompt
    
    def generate(
        self,
        question: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        mode: str = "adaptive_retrieval"
    ) -> Dict[str, Any]:
        """
        Generate answer using Self-RAG.
        
        Args:
            question: The input question
            retrieved_docs: List of retrieved documents (optional)
            mode: One of ['adaptive_retrieval', 'no_retrieval', 'always_retrieve']
            
        Returns:
            Dictionary containing:
                - answer: Generated answer
                - retrieval_decision: Whether retrieval was used
                - scores: Relevance, groundedness, utility scores
        """
        self._initialize_model()
        
        if self.model is None:
            # Fallback: simple generation without Self-RAG
            warnings.warn("vllm not available. Using fallback mode.")
            return {
                "answer": "Self-RAG not available",
                "retrieval_decision": False,
                "scores": {}
            }
        
        # Step 1: Generate without retrieval to check if retrieval is needed
        prompt = self.format_prompt(question)
        sampling_params = self.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            skip_special_tokens=False,
            logprobs=5000
        )
        
        # Determine if retrieval is needed
        if mode == "always_retrieve":
            do_retrieve = True
        elif mode == "no_retrieval":
            do_retrieve = False
        else:  # adaptive_retrieval
            # Generate initial response to check retrieval token
            preds = self.model.generate([prompt], sampling_params)
            pred_log_probs = preds[0].outputs[0].logprobs
            
            if len(pred_log_probs) > 0:
                score_dict = {}
                for tok, id in self.ret_tokens.items():
                    if id not in pred_log_probs[0]:
                        score_dict[tok] = -100
                    else:
                        prob = pred_log_probs[0][id]
                        score_dict[tok] = float(prob)
                
                # Decide based on threshold
                ret_score = score_dict["[Retrieval]"]
                no_ret_score = score_dict["[No Retrieval]"]
                do_retrieve = ret_score / (ret_score + no_ret_score) > self.threshold
            else:
                do_retrieve = False
        
        # Step 2: Generate with or without retrieval
        if do_retrieve and retrieved_docs and len(retrieved_docs) > 0:
            # Generate with each retrieved document
            prompts = []
            for doc in retrieved_docs[:5]:  # Top 5 docs
                context = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                title = doc.get('title', '') if isinstance(doc, dict) else ''
                if title:
                    context = f"{title}\n{context}"
                prompts.append(self.format_prompt(question, context))
            
            # Generate for all contexts
            preds = self.model.generate(prompts, sampling_params)
            
            # Score and select best output
            best_output = None
            best_score = -float('inf')
            
            for pred in preds:
                pred_text = self._postprocess(pred.outputs[0].text)
                pred_token_ids = pred.outputs[0].token_ids
                pred_log_probs = pred.outputs[0].logprobs
                
                # Compute scores
                scores = self._compute_scores(pred_token_ids, pred_log_probs)
                
                # Compute overall score
                overall_score = (
                    self.w_rel * scores.get('relevance', 0.0) +
                    self.w_sup * scores.get('groundedness', 0.0) +
                    self.w_use * scores.get('utility', 0.0)
                )
                
                if self.use_seqscore:
                    seq_score = pred.outputs[0].cumulative_logprob / max(len(pred_token_ids), 1)
                    overall_score += seq_score
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_output = {
                        "answer": pred_text,
                        "retrieval_decision": True,
                        "scores": scores,
                        "overall_score": overall_score
                    }
            
            return best_output if best_output else {
                "answer": "Error in generation",
                "retrieval_decision": True,
                "scores": {}
            }
        else:
            # Generate without retrieval
            preds = self.model.generate([prompt], sampling_params)
            pred_text = self._postprocess(preds[0].outputs[0].text)
            
            return {
                "answer": pred_text,
                "retrieval_decision": False,
                "scores": {}
            }
    
    def _compute_scores(
        self,
        token_ids: List[int],
        log_probs: List[Dict[int, float]]
    ) -> Dict[str, float]:
        """Compute relevance, groundedness, and utility scores."""
        scores = {}
        
        # Relevance score
        if len(log_probs) > 0 and self.rel_tokens:
            rel_scores = {}
            for tok, id in self.rel_tokens.items():
                prob = log_probs[0].get(id, -100)
                rel_scores[tok] = np.exp(float(prob))
            
            total = sum(rel_scores.values())
            if total > 0:
                scores['relevance'] = rel_scores["[Relevant]"] / total
        
        # Groundedness score
        if self.use_groundness and self.grd_tokens:
            grd_scores = {}
            for idx, tok_id in enumerate(token_ids):
                if tok_id in self.grd_tokens.values():
                    for tok, id in self.grd_tokens.items():
                        prob = log_probs[idx].get(id, -100)
                        grd_scores[tok] = np.exp(float(prob))
                    break
            
            if len(grd_scores) == 3:
                total = sum(grd_scores.values())
                scores['groundedness'] = (
                    grd_scores["[Fully supported]"] +
                    0.5 * grd_scores["[Partially supported]"]
                ) / total if total > 0 else 0.0
        
        # Utility score
        if self.use_utility and self.ut_tokens:
            ut_scores = {}
            for idx, tok_id in enumerate(token_ids):
                if tok_id in self.ut_tokens.values():
                    for tok, id in self.ut_tokens.items():
                        prob = log_probs[idx].get(id, -100)
                        ut_scores[tok] = np.exp(float(prob))
                    break
            
            if len(ut_scores) == 5:
                total = sum(ut_scores.values())
                ut_values = [-1, -0.5, 0, 0.5, 1]
                scores['utility'] = sum(
                    ut_values[i] * (ut_scores[f"[Utility:{i+1}]"] / total)
                    for i in range(5)
                ) if total > 0 else 0.0
        
        return scores
    
    def _postprocess(self, text: str) -> str:
        """Remove control tokens from generated text."""
        for token in CONTROL_TOKENS:
            text = text.replace(token, "")
        
        # Remove special markers
        text = text.replace("</s>", "").replace("<|endoftext|>", "")
        text = text.replace("<paragraph>", "").replace("</paragraph>", "")
        text = text.strip()
        
        return text
    
    def __repr__(self):
        return f"SelfRAG(model={self.model_name}, threshold={self.threshold})"

