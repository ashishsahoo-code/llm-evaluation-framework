# LLM-as-a-Judge Evaluation Framework

A small, extensible Python framework for evaluating LLM-generated answers across relevance, faithfulness, and reference-answer overlap.

## Features
- Typed Pydantic evaluation contracts
- Pluggable judge interface
- Dependency-free lexical baseline judge
- Optional OpenAI judge adapter
- JSONL datasets and JSON reports
- Per-case and aggregate scores
- Unit tests and GitHub Actions CI

> The heuristic judge is a transparent baseline, not a semantic evaluator or benchmark-quality metric. Validate LLM-judge outputs against human annotations before using them for release decisions.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
llm-eval run --input examples/sample_dataset.jsonl --output reports/evaluation.json
```

For the OpenAI judge:
```bash
export OPENAI_API_KEY="your-key"
llm-eval run --input examples/sample_dataset.jsonl --output reports/openai.json --judge openai --model gpt-4o-mini
```

## Dataset format
One JSON object per line:
```json
{"case_id":"refund-001","question":"What is the refund period?","answer":"30 days.","context":"Refunds are available within 30 days.","reference_answer":"Customers may request a refund within 30 days."}
```

## Architecture
```text
JSONL dataset -> EvaluationRunner -> Judge adapter -> Case scores -> JSON report
                                  |-> HeuristicJudge
                                  |-> OpenAIJudge
```

## Evaluation dimensions
- **Relevance:** Does the answer address the question?
- **Faithfulness:** Is the answer supported by supplied context?
- **Reference overlap:** Does it agree with the optional reference answer?

Missing evidence is marked `warn` and excluded from the average rather than fabricated as a score.

## Security notes
- Treat dataset content as untrusted input.
- Keep API keys in environment variables; never commit secrets.
- The OpenAI adapter explicitly instructs the judge not to follow instructions embedded in evaluated content.
- Review judge rationales and calibrate thresholds for your use case.

## Development
```bash
pytest
ruff check src tests
```

## Roadmap
- Human-annotated calibration sets
- Pairwise ranking and rubric-based evaluators
- Additional provider adapters
- Bootstrap confidence intervals and judge agreement analysis

## License
MIT
