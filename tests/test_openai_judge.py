import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_eval.schemas import EvaluationCase


def load_judge(monkeypatch, response_content):
    calls = []

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        def create_completion(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=response_content)
                    )
                ]
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    module = importlib.import_module("llm_eval.judges.openai_judge")
    module = importlib.reload(module)
    return module.OpenAIJudge(api_key="test-key", model="test-model"), calls


def sample_case():
    return EvaluationCase(
        case_id="judge-001",
        question="What is the refund period?",
        answer="Refunds are available within 30 days.",
        context="Customers can request refunds within 30 days.",
        reference_answer="Refunds are available within 30 days.",
    )


def valid_response():
    return json.dumps(
        {
            "scores": [
                {
                    "name": "relevance",
                    "score": 0.9,
                    "reasoning": "The answer addresses the question.",
                    "verdict": "pass",
                }
            ]
        }
    )


def test_openai_judge_parses_valid_response(monkeypatch):
    judge, _ = load_judge(monkeypatch, valid_response())

    scores = judge.evaluate(sample_case())

    assert len(scores) == 1
    assert scores[0].name == "relevance"
    assert scores[0].score == 0.9
    assert scores[0].verdict == "pass"


def test_openai_judge_sends_expected_model_and_prompt(monkeypatch):
    judge, calls = load_judge(monkeypatch, valid_response())

    judge.evaluate(sample_case())

    assert judge.model_name == "test-model"
    assert len(calls) == 1
    request = calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["role"] == "system"
    assert "untrusted data" in request["messages"][0]["content"]
    assert json.loads(request["messages"][1]["content"])["case_id"] == "judge-001"


def test_openai_judge_rejects_malformed_json(monkeypatch):
    judge, _ = load_judge(monkeypatch, "{not-valid-json")

    with pytest.raises(json.JSONDecodeError):
        judge.evaluate(sample_case())


def test_openai_judge_rejects_score_outside_range(monkeypatch):
    response = json.dumps(
        {
            "scores": [
                {
                    "name": "relevance",
                    "score": 1.5,
                    "reasoning": "Invalid score.",
                    "verdict": "pass",
                }
            ]
        }
    )
    judge, _ = load_judge(monkeypatch, response)

    with pytest.raises(ValidationError):
        judge.evaluate(sample_case())
