import sys
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import HelpCentreAssistant
from src.utils import get_openai_llm
from src.models.eval import AnswerEval


JUDGE_SYS = (
    "You are a strict evaluator. Score fields in [0,1] and set verdict to 'pass' or 'fail'."
)

JUDGE_TMPL = (
    "Question: {q}\n\n"
    "System answer:\n```text\n{ans}\n```\n\n"
    "Reference answer (may be partial):\n```text\n{ref}\n```\n\n"
    "Score: correctness (solves question), faithfulness (supported by docs), relevance (addresses question)."
)


def timestamp_dir(base: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = base / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_synthetic.jsonl")
    parser.add_argument("--judge_model", default=os.getenv("ANSWER_JUDGE_MODEL", "gpt-4.1"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    run_dir = timestamp_dir(Path("data/eval_runs"))
    out_file = run_dir / "llm_eval.json"

    assistant = HelpCentreAssistant()
    judge = get_openai_llm(model=args.judge_model, temperature=0).with_structured_output(AnswerEval)

    records = [json.loads(l) for l in ds_path.read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    results: List[Dict] = []

    num_records = len(records)
    for idx, rec in enumerate(records, start=1):
        print(f"[{idx}/{num_records}] Query: {rec['question']}...", flush=True)
        q = rec["question"]
        ans = assistant.run(q, thread_id=f"eval-{rec['id']}")
        try:
            s = judge.invoke([
                {"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": JUDGE_TMPL.format(q=q, ans=ans, ref=rec.get("reference_answer", ""))},
            ])
            res = {
                "id": rec.get("id"),
                "type": rec.get("type", ""),
                "correctness": float(s.correctness),
                "faithfulness": float(s.faithfulness),
                "relevance": float(s.relevance),
                "verdict": s.verdict,
            }
        except Exception:
            res = {
                "id": rec.get("id"),
                "type": rec.get("type", ""),
                "correctness": 0.0,
                "faithfulness": 0.0,
                "relevance": 0.0,
                "verdict": "fail",
            }
        results.append(res)

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "avg_correctness": avg([x["correctness"] for x in results]),
        "avg_faithfulness": avg([x["faithfulness"] for x in results]),
        "avg_relevance": avg([x["relevance"] for x in results]),
        "pass_rate": avg([1.0 if x["verdict"] == "pass" else 0.0 for x in results]),
        "n": len(results),
        "by_type": {},
    }

    types = sorted(set(x.get("type", "") for x in results))
    for t in types:
        subset = [x for x in results if x.get("type", "") == t]
        summary["by_type"][t] = {
            "n": len(subset),
            "pass_rate": avg([1.0 if x["verdict"] == "pass" else 0.0 for x in subset]),
        }

    out = {"summary": summary, "details": results}
    out_file.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

