from collections import defaultdict
from uuid import uuid4

from llm_eval.judges.base import Judge
from llm_eval.schemas import CaseEvaluation, EvaluationCase, EvaluationReport


class EvaluationRunner:
    def __init__(self, judge: Judge, pass_threshold: float = 0.70):
        if not 0 <= pass_threshold <= 1:
            raise ValueError("pass_threshold must be between 0 and 1")
        self.judge, self.pass_threshold = judge, pass_threshold

    def evaluate_case(self, case: EvaluationCase) -> CaseEvaluation:
        scores = self.judge.evaluate(case)
        if not scores:
            raise ValueError(f"Judge returned no scores for case {case.case_id}")
        available = [s.score for s in scores if s.verdict != "warn"]
        overall = sum(available) / len(available) if available else 0.0
        warnings = [s.name for s in scores if s.verdict == "warn"]
        return CaseEvaluation(
            case_id=case.case_id, scores=scores, overall_score=round(overall, 4),
            passed=overall >= self.pass_threshold and not any(s.verdict == "fail" for s in scores),
            judge_model=self.judge.model_name, warnings=warnings,
        )

    def run(self, cases: list[EvaluationCase]) -> EvaluationReport:
        if not cases:
            raise ValueError("At least one evaluation case is required")
        results = [self.evaluate_case(case) for case in cases]
        grouped = defaultdict(list)
        for result in results:
            for score in result.scores:
                if score.verdict != "warn":
                    grouped[score.name].append(score.score)
        averages = {key: round(sum(vals) / len(vals), 4) for key, vals in grouped.items() if vals}
        passed = sum(result.passed for result in results)
        return EvaluationReport(
            run_id=str(uuid4()), total_cases=len(results), passed_cases=passed,
            failed_cases=len(results) - passed, pass_rate=round(passed / len(results), 4),
            average_score=round(sum(r.overall_score for r in results) / len(results), 4),
            dimension_averages=averages, results=results,
        )
