import typer
from llm_eval.dataset import load_jsonl
from llm_eval.judges.heuristic import HeuristicJudge
from llm_eval.reporting import write_json
from llm_eval.runner import EvaluationRunner

app = typer.Typer(help="Evaluate LLM responses with pluggable judges.")

@app.command()
def run(
    input: str = typer.Option(..., "--input", help="Path to JSONL dataset"),
    output: str = typer.Option("reports/evaluation.json", "--output"),
    judge: str = typer.Option("heuristic", "--judge"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    threshold: float = typer.Option(0.70, "--threshold"),
):
    if judge == "heuristic":
        evaluator = HeuristicJudge()
    elif judge == "openai":
        try:
            from llm_eval.judges.openai_judge import OpenAIJudge
        except ImportError as exc:
            raise typer.BadParameter("Install optional dependency: pip install -e '.[openai]'") from exc
        evaluator = OpenAIJudge(model=model)
    else:
        raise typer.BadParameter("judge must be 'heuristic' or 'openai'")
    report = EvaluationRunner(evaluator, threshold).run(load_jsonl(input))
    write_json(report, output)
    typer.echo(f"Evaluated {report.total_cases} cases; pass rate={report.pass_rate:.1%}; report={output}")

if __name__ == "__main__":
    app()
