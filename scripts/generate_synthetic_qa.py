import argparse
import math
import json
import os
import uuid
from pathlib import Path
from typing import List

import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_openai_llm
from src.models.eval import QAList, UnanswerableQuestion


GEN_SYS = "You generate Q/A pairs strictly from the provided documentation chunk. Do not add outside knowledge."
GEN_PROMPT = (
    "Create 2 Q/A items covering a fact lookup and a how-to (steps).\n"
    "- Each answer must be fully supported by the chunk.\n"
    "- Quote 1-3 verbatim support spans from the chunk for each item.\n"
    "Return using the provided schema.\n\n"
    "Chunk:\n```text\n{chunk}\n```"
)

NEG_PROMPT = (
    "Given the chunk, produce ONE plausible question that cannot be answered from this chunk alone.\n"
    "Return using the provided schema.\n\n"
    "Chunk:\n```text\n{chunk}\n```"
)

 


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.json", help="Path to chunks JSON array")
    parser.add_argument("--out", default=None, help="Path to output JSON (defaults to data/eval_synthetic.json)")
    parser.add_argument("--model", default=os.getenv("GEN_MODEL", "gpt-4.1"))
    parser.add_argument("--max_items", type=int, default=3, help="Max QA items per chunk")
    parser.add_argument("--max_chunks", type=int, default=None, help="Limit number of chunks to process")
    parser.add_argument("--neg_ratio", type=float, default=0.1, help="Unanswerables per answerable (e.g., 0.1 = 10%)")
    parser.add_argument("--neg_per_chunk", type=int, default=None, help="Override to generate a fixed number per chunk")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    default_out = Path("data/eval_synthetic.json")
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[generate_synthetic_qa] Arguments:", flush=True)
    print(f"  chunks: {chunks_path}", flush=True)
    print(f"  out: {out_path}", flush=True)
    print(f"  model: {args.model}", flush=True)
    print(f"  max_items: {args.max_items}", flush=True)
    print(f"  max_chunks: {args.max_chunks}", flush=True)
    print(f"  neg_ratio: {args.neg_ratio}", flush=True)
    print(f"  neg_per_chunk: {args.neg_per_chunk}", flush=True)


    llm = get_openai_llm(model=args.model)
    qa_struct = llm.with_structured_output(QAList)
    neg_struct = llm.with_structured_output(UnanswerableQuestion)

    print("[generate_synthetic_qa] Initialized LLM and structured outputs", flush=True)

    chunks: List[str] = json.loads(chunks_path.read_text())
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    print(f"[generate_synthetic_qa] Loaded {len(chunks)} chunks", flush=True)

    written = 0
    with out_path.open("w") as fout:
        for cid, chunk in enumerate(chunks):
            print(f"[generate_synthetic_qa] Processing chunk {cid + 1}/{len(chunks)}", flush=True)
            chunk_trim = chunk[:4000]
            print(f"  chunk length: original={len(chunk)} trimmed={len(chunk_trim)}", flush=True)

            n_pos = 0
            try:
                print("  Generating positive Q/A items...", flush=True)
                resp = qa_struct.invoke([
                    {"role": "system", "content": GEN_SYS},
                    {"role": "user", "content": GEN_PROMPT.format(chunk=chunk_trim)},
                ])
                num_items = len(resp.items)
                print(f"  Model returned {num_items} items; writing up to {args.max_items}", flush=True)
                for item in resp.items[: args.max_items]:
                    rec = {
                        "id": str(uuid.uuid4()),
                        "question": item.question,
                        "reference_answer": item.answer,
                        "source_chunk_ids": [cid],
                        "url": None,
                        "split": "test",
                        "type": "fact" if "?" in item.question else "howto",
                        "evidence_spans": item.evidence_spans,
                    }
                    line = json.dumps(rec)
                    fout.write(line + "\n")
                    written += 1
                    n_pos += 1
                print(f"  Wrote {n_pos} positive items", flush=True)
            except Exception as e:
                print(f"  Error generating positives for chunk {cid}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

            neg_count = args.neg_per_chunk if args.neg_per_chunk is not None else (math.ceil(n_pos * args.neg_ratio) if n_pos > 0 else 0)
            print(f"  Generating {neg_count} unanswerable question(s)", flush=True)
            for i in range(max(0, neg_count)):
                try:
                    neg = neg_struct.invoke([
                        {"role": "system", "content": GEN_SYS},
                        {"role": "user", "content": NEG_PROMPT.format(chunk=chunk_trim)},
                    ])
                    recu = {
                        "id": str(uuid.uuid4()),
                        "question": neg.question.strip(),
                        "reference_answer": "",
                        "source_chunk_ids": [],
                        "url": None,
                        "split": "test",
                        "type": "unanswerable",
                        "evidence_spans": [],
                    }
                    line = json.dumps(recu)
                    fout.write(line + "\n")
                    written += 1
                except Exception as e:
                    print(f"  Error generating unanswerable {i+1}/{neg_count} for chunk {cid}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()

    print(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    main()

