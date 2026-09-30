"""Stage-level retrieval diagnostics using the same retrieval primitives as evaluate_retrieval."""
import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer

from scripts.evaluate_retrieval import bm25_search, dense_search, percentile, resolve_model, rerank, rrf


def unique_rows(rows):
    result = []
    seen = set()
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            result.append(row)
    return result


def candidate_hash(rows):
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode("utf-8")).hexdigest()[:16]


def rank_map(rows):
    return {row["id"]: index for index, row in enumerate(rows, 1)}


def ndcg_at_k(retrieved, relevant, k=10):
    if not relevant:
        return None
    dcg = sum(1 / math.log2(index + 2) for index, doc_id in enumerate(retrieved[:k]) if doc_id in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def stage_metrics(query_rows, stage_name, k=10):
    eligible = [row for row in query_rows if row["relevant_doc_ids"]]
    recalls = []
    hits = []
    mrrs = []
    ndcgs = []
    for row in eligible:
        relevant = set(row["relevant_doc_ids"])
        ids = row["stages"][stage_name]["ids"][:k]
        hit_ids = set(ids) & relevant
        recalls.append(len(hit_ids) / len(relevant))
        hits.append(float(bool(hit_ids)))
        first = next((index for index, doc_id in enumerate(ids, 1) if doc_id in relevant), None)
        mrrs.append(1 / first if first else 0.0)
        ndcgs.append(ndcg_at_k(ids, relevant, k))
    return {
        f"Recall@{k}": statistics.mean(recalls) if recalls else 0.0,
        f"HitRate@{k}": statistics.mean(hits) if hits else 0.0,
        f"MRR@{k}": statistics.mean(mrrs) if mrrs else 0.0,
        f"nDCG@{k}": statistics.mean(ndcgs) if ndcgs else 0.0,
        "eligible_queries": len(eligible),
        "no_relevant_queries": len(query_rows) - len(eligible),
    }


def candidate_coverage(query_rows, stage_name):
    eligible = [row for row in query_rows if row["relevant_doc_ids"]]
    recalls = []
    hits = []
    sizes = []
    for row in eligible:
        relevant = set(row["relevant_doc_ids"])
        ids = set(row["stages"][stage_name]["ids"])
        recalls.append(len(ids & relevant) / len(relevant))
        hits.append(float(bool(ids & relevant)))
        sizes.append(row["stages"][stage_name]["count"])
    return {"macro_recall": statistics.mean(recalls) if recalls else 0.0,
            "hit_rate": statistics.mean(hits) if hits else 0.0,
            "mean_candidate_count": statistics.mean(sizes) if sizes else 0.0,
            "p95_candidate_count": percentile(sizes, .95) if sizes else 0.0}


def make_stage(rows, relevant, fields):
    ids = [row["id"] for row in rows]
    return {
        "count": len(rows), "ids": ids, "candidate_hash": candidate_hash(rows),
        "documents": [{
            "doc_id": row["id"], "parent_doc_id": row.get("parent_id") or row.get("document_id"),
            "is_relevant": row["id"] in relevant,
            **{name: row.get(name) for name in fields},
        } for row in rows],
    }


def classify_losses(case, known_ids):
    relevant = set(case["relevant_doc_ids"])
    stages = case["stages"]
    dense = set(stages["dense_top100"]["ids"])
    sparse = set(stages["bm25_top100"]["ids"])
    union = dense | sparse
    rrf_ids = set(stages["rrf_candidates"]["ids"])
    final_ids = set(stages["final_top10"]["ids"])
    labels = {}
    for doc_id in relevant:
        if doc_id not in known_ids:
            labels[doc_id] = "not_in_index"
        elif doc_id not in union:
            labels[doc_id] = "initial_recall_miss"
        elif doc_id not in rrf_ids:
            labels[doc_id] = "rrf_truncation"
        elif doc_id not in final_ids:
            labels[doc_id] = "rerank_or_final_cut"
        else:
            labels[doc_id] = "returned"
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--bm25-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--rrf-candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--parallel-channels", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--type-boost", type=float, default=0.05)
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()
    cases = [json.loads(line) for line in open(args.dataset, encoding="utf-8") if line.strip()]
    if args.split:
        cases = [case for case in cases if case.get("split") == args.split]
    connections.connect(host=os.getenv("MILVUS_HOST", "localhost"), port=os.getenv("MILVUS_PORT", "19530"))
    if not utility.has_collection(args.collection):
        raise SystemExit("collection not found")
    collection = Collection(args.collection)
    collection.load()
    model = SentenceTransformer(resolve_model(args.model), cache_folder="./model_cache", local_files_only=True)
    bm25 = sqlite3.connect(args.bm25_index, check_same_thread=False)
    known_ids = {row[0] for row in bm25.execute("SELECT doc_id FROM documents")}
    results = []
    for case in cases:
        relevant = set(case["relevant_doc_ids"])
        started = time.perf_counter()

        def run_dense():
            begin = time.perf_counter()
            query_vector = model.encode(case["query"], normalize_embeddings=True)
            embedding_ms = (time.perf_counter() - begin) * 1000
            search_begin = time.perf_counter()
            rows = unique_rows(dense_search(collection, query_vector.tolist(), args.depth, args.nprobe))
            return query_vector, rows, embedding_ms, (time.perf_counter() - search_begin) * 1000

        def run_sparse():
            begin = time.perf_counter()
            rows = unique_rows(bm25_search(bm25, case["query"], args.depth))
            return rows, (time.perf_counter() - begin) * 1000

        if args.parallel_channels:
            with ThreadPoolExecutor(max_workers=2) as pool:
                dense_future = pool.submit(run_dense)
                sparse_future = pool.submit(run_sparse)
                query_vector, dense_rows, embedding_ms, dense_ms = dense_future.result()
                sparse_rows, bm25_ms = sparse_future.result()
        else:
            query_vector, dense_rows, embedding_ms, dense_ms = run_dense()
            sparse_rows, bm25_ms = run_sparse()
        initial_wall_ms = (time.perf_counter() - started) * 1000
        fusion_started = time.perf_counter()
        union_rows = unique_rows(dense_rows + sparse_rows)
        rrf_rows = rrf(dense_rows, sparse_rows, args.rrf_candidate_k, args.rrf_k)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        rerank_started = time.perf_counter()
        reranked_all = (list(rrf_rows) if args.skip_rerank else
                        rerank(model, case["query"], np.asarray(query_vector), rrf_rows, len(rrf_rows), args.type_boost))
        rerank_ms = 0.0 if args.skip_rerank else (time.perf_counter() - rerank_started) * 1000
        final_rows = reranked_all[:10]
        stages = {
            "dense_top100": make_stage(dense_rows, relevant, ("score",)),
            "bm25_top100": make_stage(sparse_rows, relevant, ("score",)),
            "union": make_stage(union_rows, relevant, ()),
            "rrf_candidates": make_stage(rrf_rows, relevant, ("score", "sources")),
            "rrf_top10": make_stage(rrf_rows[:10], relevant, ("score", "sources")),
            "rerank_full": make_stage(reranked_all, relevant, ("score", "sources")),
            "final_top10": make_stage(final_rows, relevant, ("score", "sources")),
        }
        result = {
            "query_id": case["query_id"], "query": case["query"], "category": case.get("category"),
            "relevant_doc_ids": case["relevant_doc_ids"], "stages": stages,
            "latency_ms": {"query_embedding": embedding_ms, "dense_search": dense_ms, "bm25_search": bm25_ms,
                "fusion_dedup": fusion_ms, "rerank_inference": rerank_ms,
                "initial_retrieval_wall": initial_wall_ms, "total_wall": (time.perf_counter() - started) * 1000},
        }
        result["losses"] = classify_losses(result, known_ids)
        results.append(result)
    loss_counts = {}
    for result in results:
        for label in result["losses"].values():
            loss_counts[label] = loss_counts.get(label, 0) + 1
    latency = {name: [row["latency_ms"][name] for row in results] for name in results[0]["latency_ms"]}
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(), "collection": args.collection,
        "entity_count": collection.num_entities, "bm25_index": args.bm25_index, "bm25_id_count": len(known_ids),
        "split": args.split, "query_count": len(results), "depth": args.depth,
        "rrf_candidate_k": args.rrf_candidate_k, "rrf_k": args.rrf_k, "nprobe": args.nprobe,
        "parallel_channels": args.parallel_channels, "skip_rerank": args.skip_rerank, "type_boost": args.type_boost,
        "metrics": {name: stage_metrics(results, name) for name in ("dense_top100", "bm25_top100", "union", "rrf_candidates", "rrf_top10", "final_top10")},
        "candidate_coverage": {name: candidate_coverage(results, name) for name in ("dense_top100", "bm25_top100", "union", "rrf_candidates", "rerank_full")},
        "oracle_recall_at_10": statistics.mean(min(10, len(set(row["stages"]["rrf_candidates"]["ids"]) & set(row["relevant_doc_ids"]))) / len(row["relevant_doc_ids"]) for row in results if row["relevant_doc_ids"]),
        "loss_counts": loss_counts,
        "latency_summary_ms": {name: {"mean": statistics.mean(values), "p50": percentile(values, .5), "p95": percentile(values, .95)} for name, values in latency.items()},
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(json.dumps({"metrics": output["metrics"], "oracle_recall_at_10": output["oracle_recall_at_10"], "loss_counts": loss_counts, "latency_summary_ms": output["latency_summary_ms"]}, ensure_ascii=False, indent=2))
    bm25.close()
    connections.disconnect("default")


if __name__ == "__main__":
    main()
