import json
from pathlib import Path

from llm_eval.schemas import EvaluationReport


def write_json(report: EvaluationReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
