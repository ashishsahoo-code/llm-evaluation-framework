from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    question: str
    answer: str
    context: str = ""
    reference_answer: str | None = None

class DimensionScore(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    verdict: Literal["pass", "warn", "fail"]

class CaseEvaluation(BaseModel):
    case_id: str
    scores: list[DimensionScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    judge_model: str
    warnings: list[str] = Field(default_factory=list)

class EvaluationReport(BaseModel):
    run_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    average_score: float
    dimension_averages: dict[str, float]
    results: list[CaseEvaluation]
