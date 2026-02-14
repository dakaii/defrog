"""
RAG Evaluation Module
Implements RAGAS-style metrics for evaluating RAG pipeline quality
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    query: str
    answer: str
    contexts: List[str]
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    overall_score: float
    metadata: Dict
    timestamp: str


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using LLM-based metrics
    Inspired by RAGAS framework
    """
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        """Initialize evaluator with LLM client"""
        if llm_client:
            self.client = llm_client
        else:
            api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('LLM_BASE_URL')
            
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        
        self.model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the answer is faithful to the retrieved contexts
        Score: 0.0 (hallucination) to 1.0 (fully grounded)
        """
        context_text = "\n\n".join(contexts)
        
        prompt = f"""You are an expert evaluator assessing RAG system faithfulness.
        
Given the following contexts and answer, evaluate if EVERY claim in the answer 
is directly supported by the contexts. 

Contexts:
{context_text}

Answer:
{answer}

For each statement in the answer:
1. Check if it's directly supported by the contexts
2. Mark as "supported" or "unsupported"

Output format:
- List each claim/statement
- Mark as SUPPORTED or UNSUPPORTED
- Final score: (supported claims / total claims)

Be strict - implications and inferences count as UNSUPPORTED unless explicitly stated."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            evaluation = response.choices[0].message.content
            
            # Extract score from response
            if "Final score:" in evaluation:
                score_text = evaluation.split("Final score:")[-1].strip()
                # Try to parse fraction like "3/4" or decimal like "0.75"
                if "/" in score_text:
                    nums = score_text.split("/")
                    score = float(nums[0].strip()) / float(nums[1].split()[0].strip())
                else:
                    # Extract first number found
                    import re
                    numbers = re.findall(r'[\d.]+', score_text)
                    score = float(numbers[0]) if numbers else 0.5
            else:
                score = 0.5  # Default if parsing fails
            
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return 0.5
    
    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """
        Evaluate if the answer is relevant to the query
        Score: 0.0 (irrelevant) to 1.0 (perfectly relevant)
        """
        prompt = f"""You are an expert evaluator assessing answer relevancy.
        
Question: {query}
Answer: {answer}

Evaluate how well the answer addresses the question:
1. Does it answer the core question?
2. Is it complete?
3. Is it focused (not too much irrelevant information)?

Rate on a scale of 0-100 where:
- 0-20: Completely irrelevant or wrong topic
- 20-40: Somewhat related but misses the main point
- 40-60: Partially answers the question
- 60-80: Good answer with minor gaps
- 80-100: Excellent, complete, and focused answer

Output your reasoning and then state: "SCORE: [number]" """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            evaluation = response.choices[0].message.content
            
            # Extract score
            import re
            score_match = re.search(r'SCORE:\s*(\d+)', evaluation)
            if score_match:
                score = float(score_match.group(1)) / 100.0
            else:
                score = 0.5
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Answer relevancy evaluation failed: {e}")
            return 0.5
    
    def evaluate_context_recall(self, query: str, ground_truth: str, contexts: List[str]) -> float:
        """
        Evaluate if retrieved contexts contain information needed for ground truth
        Score: 0.0 (missing info) to 1.0 (all info present)
        """
        context_text = "\n\n".join(contexts)
        
        prompt = f"""You are evaluating context recall for a RAG system.

Question: {query}
Expected Answer: {ground_truth}

Retrieved Contexts:
{context_text}

Evaluate: Do the retrieved contexts contain all the information needed to generate the expected answer?

For each key piece of information in the expected answer:
1. Check if it's present in the contexts
2. Mark as FOUND or MISSING

Output:
- List each key information piece
- Mark as FOUND or MISSING  
- Final score: (found pieces / total pieces)"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            evaluation = response.choices[0].message.content
            
            # Count FOUND vs MISSING
            found_count = evaluation.count("FOUND")
            missing_count = evaluation.count("MISSING")
            total = found_count + missing_count
            
            if total > 0:
                score = found_count / total
            else:
                score = 0.5
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Context recall evaluation failed: {e}")
            return 0.5
    
    def evaluate_context_precision(self, query: str, contexts: List[str]) -> float:
        """
        Evaluate if retrieved contexts are precise and not noisy
        Score: 0.0 (all noise) to 1.0 (all relevant)
        """
        prompt = f"""You are evaluating context precision for a RAG system.

Question: {query}

Rate each context for relevance to the question (0=irrelevant, 1=highly relevant):

"""
        scores = []
        
        for i, context in enumerate(contexts, 1):
            prompt += f"\nContext {i}:\n{context[:500]}...\n"
        
        prompt += "\nFor each context, provide a relevance score 0-100."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            evaluation = response.choices[0].message.content
            
            # Extract scores
            import re
            scores = re.findall(r'\d+', evaluation)
            scores = [min(float(s) / 100.0, 1.0) for s in scores[:len(contexts)]]
            
            if scores:
                # Weight earlier contexts more heavily (assuming ranking order)
                weights = [1.0 / (i + 1) for i in range(len(scores))]
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                return weighted_score
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Context precision evaluation failed: {e}")
            return 0.5
    
    def evaluate(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Run complete evaluation suite
        """
        logger.info(f"Evaluating query: {query[:50]}...")
        
        # Run evaluations in parallel for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            faithfulness_future = executor.submit(
                self.evaluate_faithfulness, answer, contexts
            )
            relevancy_future = executor.submit(
                self.evaluate_answer_relevancy, query, answer
            )
            precision_future = executor.submit(
                self.evaluate_context_precision, query, contexts
            )
            
            # Context recall only if ground truth is provided
            if ground_truth:
                recall_future = executor.submit(
                    self.evaluate_context_recall, query, ground_truth, contexts
                )
                context_recall = recall_future.result()
            else:
                context_recall = None
            
            faithfulness = faithfulness_future.result()
            answer_relevancy = relevancy_future.result()
            context_precision = precision_future.result()
        
        # Calculate overall score
        scores = [faithfulness, answer_relevancy, context_precision]
        if context_recall is not None:
            scores.append(context_recall)
        overall_score = np.mean(scores)
        
        result = EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall if context_recall is not None else -1,
            context_precision=context_precision,
            overall_score=overall_score,
            metadata={
                "model": self.model,
                "num_contexts": len(contexts),
                "has_ground_truth": ground_truth is not None
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Evaluation complete. Overall score: {overall_score:.2f}")
        return result
    
    def evaluate_batch(
        self, 
        test_cases: List[Dict]
    ) -> Tuple[List[EvaluationResult], Dict]:
        """
        Evaluate multiple test cases and provide aggregate metrics
        
        test_cases: List of dicts with keys: query, answer, contexts, ground_truth (optional)
        """
        results = []
        
        for case in test_cases:
            result = self.evaluate(
                query=case['query'],
                answer=case['answer'],
                contexts=case['contexts'],
                ground_truth=case.get('ground_truth')
            )
            results.append(result)
        
        # Calculate aggregate metrics
        aggregate = {
            'num_evaluations': len(results),
            'avg_faithfulness': np.mean([r.faithfulness for r in results]),
            'avg_answer_relevancy': np.mean([r.answer_relevancy for r in results]),
            'avg_context_precision': np.mean([r.context_precision for r in results]),
            'avg_overall_score': np.mean([r.overall_score for r in results]),
            'std_overall_score': np.std([r.overall_score for r in results])
        }
        
        # Add context recall if available
        recall_scores = [r.context_recall for r in results if r.context_recall >= 0]
        if recall_scores:
            aggregate['avg_context_recall'] = np.mean(recall_scores)
        
        return results, aggregate
    
    def generate_report(self, results: List[EvaluationResult], output_path: str = "rag_evaluation_report.json"):
        """Generate evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(results),
            'results': [
                {
                    'query': r.query[:100],
                    'faithfulness': r.faithfulness,
                    'answer_relevancy': r.answer_relevancy,
                    'context_recall': r.context_recall,
                    'context_precision': r.context_precision,
                    'overall_score': r.overall_score
                }
                for r in results
            ],
            'summary': {
                'avg_faithfulness': np.mean([r.faithfulness for r in results]),
                'avg_answer_relevancy': np.mean([r.answer_relevancy for r in results]),
                'avg_context_precision': np.mean([r.context_precision for r in results]),
                'avg_overall_score': np.mean([r.overall_score for r in results]),
                'min_overall_score': min(r.overall_score for r in results),
                'max_overall_score': max(r.overall_score for r in results)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return report