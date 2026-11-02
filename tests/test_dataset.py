import pytest
from llm_eval.dataset import load_jsonl

def test_load_sample_dataset():
    cases = load_jsonl("examples/sample_dataset.jsonl")
    assert len(cases) == 3
    assert cases[0].case_id == "support-001"

def test_invalid_row_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"case_id":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_jsonl(path)
