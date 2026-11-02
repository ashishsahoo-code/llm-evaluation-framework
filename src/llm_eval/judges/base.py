from typing import Protocol

from llm_eval.schemas import DimensionScore, EvaluationCase


class Judge(Protocol):
    @property
    def model_name(self) -> str: ...
    def evaluate(self, case: EvaluationCase) -> list[DimensionScore]: ...
