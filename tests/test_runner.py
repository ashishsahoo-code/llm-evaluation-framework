import pytest

from llm_eval.judges.heuristic import HeuristicJudge
from llm_eval.runner import EvaluationRunner
from llm_eval.schemas import EvaluationCase


def sample():
    return EvaluationCase(
        case_id="1", question="What is the refund period?",
        answer="Refunds are available within 30 days.",
        context="Customers can request refunds within 30 days.",
        reference_answer="Refunds are available within 30 days.",
    )


def test_runner_aggregates_cases():
    report = EvaluationRunner(HeuristicJudge()).run([sample(), sample()])
    assert report.total_cases == 2
    assert len(report.results) == 2
    assert "relevance" in report.dimension_averages


def test_missing_context_is_warn():
    case = sample()
    case.context = ""
    result = EvaluationRunner(HeuristicJudge()).evaluate_case(case)
    faithfulness = next(s for s in result.scores if s.name == "faithfulness")
    assert faithfulness.verdict == "warn"


def test_empty_dataset_rejected():
    with pytest.raises(ValueError, match="At least one"):
        EvaluationRunner(HeuristicJudge()).run([])


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError, match="pass_threshold"):
        EvaluationRunner(HeuristicJudge(), 1.5)
