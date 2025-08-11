import os
import sys
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Set

# suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nodes.node_rag_assist import hybrid_search, rerank_results
from src.constants import RETRIEVAL_TOP_N, ALPHA, RERANKER_TOP_N


def ndcg_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Calculate NDCG@k using binary relevance (1 if in gold_ids, 0 if not)."""
    if not gold_ids:  # Avoid division by zero
        return 0.0
    
    # Create relevance array (1 for gold, 0 for non-gold)
    rel = [1 if id_ in gold_ids else 0 for id_ in retrieved_ids[:k]]
    
    # DCG calculation
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    
    # Ideal ordering would have all relevant docs first
    n_rel = min(len(gold_ids), k)  # Can't have more relevant docs than k or total gold
    ideal = [1] * n_rel + [0] * (k - n_rel)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_metrics(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> dict:
    """Calculate traditional IR metrics based on gold ID matches."""
    if not gold_ids:  # Skip metrics for queries with no gold IDs
        return {
            "hit@k": 0.0,
            "mrr@k": 0.0,
            "ndcg@k": 0.0,
            "precision@k": 0.0,
            "recall@k": 0.0
        }
    
    topk = retrieved_ids[:k]
    topk_set = set(topk)
    
    # Hit@k: 1 if any gold ID in top k
    hit = 1.0 if gold_ids & topk_set else 0.0
    
    # MRR@k: 1/rank of first relevant result
    mrr = 0.0
    for rank, id_ in enumerate(topk, start=1):
        if id_ in gold_ids:
            mrr = 1.0 / rank
            break
    
    # Precision@k: fraction of top k that are relevant
    precision = len(gold_ids & topk_set) / k
    
    # Recall@k: fraction of gold retrieved in top k
    recall = len(gold_ids & topk_set) / len(gold_ids)
    
    # NDCG@k: using binary relevance
    ndcg = ndcg_at_k(retrieved_ids, gold_ids, k)
    
    return {
        "hit@k": hit,
        "mrr@k": mrr,
        "ndcg@k": ndcg,
        "precision@k": precision,
        "recall@k": recall
    }


def timestamp_dir(base: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = base / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_synthetic.jsonl", help="Path to JSONL eval set")
    parser.add_argument("--k", type=int, default=RETRIEVAL_TOP_N)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Hybrid weight: 0.0 sparse only, 1.0 dense only")
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--reranker_top_n", type=int, default=RERANKER_TOP_N, help="Number of results to keep after reranking")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    run_dir = timestamp_dir(Path("data/eval_runs"))
    out_file = run_dir / "retrieval_eval.json"

    records = [json.loads(l) for l in ds_path.read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    # Exclude unanswerable items from retrieval evaluation
    total_records = len(records)
    records = [r for r in records if r.get("type") != "unanswerable"]
    filtered_unanswerable = total_records - len(records)

    def create_metrics_dict(k: int, stage: str):
        return {
            "hit@k": [],
            "mrr@k": [],
            "ndcg@k": [],
            "precision@k": [],
            "recall@k": [],
            "k": k,
            "stage": stage,
            "n": 0,
            "use_reranker": args.use_reranker,
            "total_records": total_records,
            "filtered_unanswerable": filtered_unanswerable,
            "alpha": args.alpha,
        }

    # Track metrics for both pre and post reranking stages
    pre_rerank_metrics = create_metrics_dict(args.k, "pre_rerank")
    post_rerank_metrics = create_metrics_dict(args.reranker_top_n, "post_rerank") if args.use_reranker else None

    detailed = []

    if not records:
        empty_metrics = create_metrics_dict(args.k, "pre_rerank")
        summary = {
            "pre_rerank": {k: 0 for k in ["hit@k", "mrr@k", "ndcg@k", "precision@k", "recall@k", "n"]} | 
                         {k: v for k, v in empty_metrics.items() if k not in ["hit@k", "mrr@k", "ndcg@k", "precision@k", "recall@k", "n"]},
            "post_rerank": None
        }
        out = {"summary": summary, "details": []}
        out_file.write_text(json.dumps(out, indent=2))
        print(json.dumps(summary, indent=2))
        return

    num_records = len(records)
    for idx, record in enumerate(records, start=1):
        q = record["question"]
        q_preview = q.replace("\n", " ")[:80]
        print(f"[{idx}/{num_records}] Query: {q_preview}...", flush=True)
        
        # Get initial hybrid search results
        res = hybrid_search(q, top_k=args.k, alpha=args.alpha)
        pre_rerank_docs = [{"id": m["id"], "text": m["metadata"]["text"]} for m in res["matches"]]
        
        # Calculate pre-rerank metrics
        pre_topk_ids = [d["id"] for d in pre_rerank_docs[:args.k]]
        gold = set(map(str, record.get("source_chunk_ids", [])))
        
        # Calculate metrics using gold IDs
        pre_metrics = calculate_metrics(pre_topk_ids, gold, args.k)
        
        # Store pre-rerank metrics
        for k in ["hit@k", "mrr@k", "ndcg@k", "precision@k", "recall@k"]:
            pre_rerank_metrics[k].append(pre_metrics[k])
        pre_rerank_metrics["n"] += 1

        # If reranker enabled, calculate post-rerank metrics
        post_metrics = None
        if args.use_reranker:
            res = rerank_results(res, q, top_k=args.reranker_top_n)
            post_docs = [{"id": d["document"]["id"], "text": d["document"]["text"]} for d in res.data]
            post_topk_ids = [d["id"] for d in post_docs[:args.reranker_top_n]]
            
            # Calculate post-rerank metrics using gold IDs
            post_metrics_dict = calculate_metrics(post_topk_ids, gold, args.reranker_top_n)
            
            # Store post-rerank metrics
            for k in ["hit@k", "mrr@k", "ndcg@k", "precision@k", "recall@k"]:
                post_rerank_metrics[k].append(post_metrics_dict[k])
            post_rerank_metrics["n"] += 1
            
            post_metrics = {
                "topk_ids": post_topk_ids,
                **post_metrics_dict
            }

        detailed.append({
            "id": record.get("id"),
            "question": q,
            "pre_rerank": {
                "topk_ids": pre_topk_ids,
                "gold": list(gold),
                **pre_metrics
            },
            "post_rerank": post_metrics
        })

    def calculate_summary(metrics):
        if not metrics or not metrics["n"]:
            return None
        return {
            "hit@k": sum(metrics["hit@k"]) / metrics["n"],
            "mrr@k": sum(metrics["mrr@k"]) / metrics["n"],
            "ndcg@k": sum(metrics["ndcg@k"]) / metrics["n"],
            "precision@k": sum(metrics["precision@k"]) / metrics["n"],
            "recall@k": sum(metrics["recall@k"]) / metrics["n"],
            **{k: v for k, v in metrics.items() if isinstance(v, (int, str, float))},
        }

    summary = {
        "pre_rerank": calculate_summary(pre_rerank_metrics),
        "post_rerank": calculate_summary(post_rerank_metrics) if args.use_reranker else None
    }

    out = {"summary": summary, "details": detailed}
    out_file.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {out_file}", flush=True)


if __name__ == "__main__":
    main()