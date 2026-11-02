import re
from llm_eval.schemas import EvaluationCase, DimensionScore


def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))

def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0

class HeuristicJudge:
    """Lexical-overlap baseline; not a semantic or benchmark-quality evaluator."""
    model_name = "heuristic-baseline"

    def evaluate(self, case: EvaluationCase) -> list[DimensionScore]:
        answer, question = tokens(case.answer), tokens(case.question)
        context, reference = tokens(case.context), tokens(case.reference_answer or "")
        values = [
            ("relevance", ratio(len(answer & question), len(question)), False),
            ("faithfulness", ratio(len(answer & context), len(answer)), not bool(case.context)),
            ("reference_overlap", ratio(len(answer & reference), len(reference)), not bool(case.reference_answer)),
        ]
        return [
            DimensionScore(
                name=name, score=round(value, 4),
                reasoning="Not scored: required evidence was not provided." if missing
                else "Lexical-overlap baseline only; inspect manually.",
                verdict="warn" if missing else ("pass" if value >= 0.6 else "fail"),
            )
            for name, value, missing in values
        ]
