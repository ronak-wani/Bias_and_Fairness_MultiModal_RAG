from ragas.metrics.collections import ExactMatch
import asyncio

class Metrics:

    def __init__(self, size):
        self.exact_match_sum = 0
        self.total_samples = 0
        self.size = size

    def exact_match(self, ground_truth, response):
        scorer = ExactMatch()
        result = asyncio.run(
            scorer.ascore(reference=ground_truth, response=response)
        )
        self.exact_match_sum += result
        print(f"Exact Match Score: {result.value}")
        eval ={
                "score": result.value,
                "sum": self.exact_match_sum,
            }
        return eval

## Positive stereotype (non-neg) and Negative stereotype (neg)