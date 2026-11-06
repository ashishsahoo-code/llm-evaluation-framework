import json

from typer.testing import CliRunner

from llm_eval.cli import app


runner = CliRunner()


def test_run_command_writes_evaluation_report(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "reports" / "evaluation.json"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "refund-001",
                        "question": "What is the refund period?",
                        "answer": "Customers can request a refund within 30 days.",
                        "context": "Refunds are available within 30 days of purchase.",
                        "reference_answer": "The refund period is 30 days.",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "shipping-001",
                        "question": "How long does shipping take?",
                        "answer": "Standard shipping takes 3 to 5 business days.",
                        "context": "Orders ship within 3 to 5 business days.",
                        "reference_answer": "Shipping takes 3 to 5 business days.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["run", "--input", str(dataset_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluated 2 cases" in result.output
    assert output_path.is_file()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["total_cases"] == 2
    assert len(report["results"]) == 2
    assert {item["case_id"] for item in report["results"]} == {
        "refund-001",
        "shipping-001",
    }
