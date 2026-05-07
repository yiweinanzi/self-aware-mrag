from collections import Counter
import re
import sys 
# sys.path.append(".") 
from .llm import LLM

DEFAULT_SYSTEM_TEMPLATE = """System:
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query and reference answer
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, but it is not as concise.

User:
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

class Evaluator:
    def __init__(self):
        self.llm = LLM("gpt-4o")
        self.system_template = DEFAULT_SYSTEM_TEMPLATE
        
    def llm_eval(self, query, reference_answer, generated_answer):
        system_prompt = self.system_template.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        try_times = 10
        while True:
            try:
                judge = self.llm.generate(query=system_prompt)
                match = re.search(r'\d', judge)
                if match:
                    score =  int(match.group(0))
                    if score >= 4:
                        passing = 1
                    else:
                        passing = 0
                    break
            except Exception as e:
                print(e)
                continue
        return score, passing, judge
    
    def evaluate(self, query, reference_answer, generated_answer):
        system_prompt = self.system_template.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        score, passing, judge = self.llm_eval(query, reference_answer, generated_answer)
        result = dict(
            score=score,
            passing=passing,
            judge=judge
        )
        return result


    
if __name__ == '__main__':
    evaluator = Evaluator()
    query = "What is the capital of France?"
    reference_answer = "The capital of France is Paris."
    generated_answer = "Paris is the capital of France."
    score, passing, judge = evaluator.llm_eval(query, reference_answer, generated_answer)
    print(score, passing, judge)
