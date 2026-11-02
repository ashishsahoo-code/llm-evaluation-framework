from pathlib import Path
import json
from llm_eval.schemas import EvaluationCase


def load_jsonl(path: str | Path) -> list[EvaluationCase]:
    cases = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid dataset row at line {line_number}: {exc}") from exc
    return cases
