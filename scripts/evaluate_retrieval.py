"""Evaluate Dense, Chinese BM25, RRF Hybrid, and local reranking under one setup."""
import argparse
import glob
import json
import os
import sqlite3
import statistics
import time
from datetime import datetime, timezone

import jieba
import numpy as np
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer

KS = (1, 3, 5, 10)
STOP_WORDS = {"的", "了", "是", "在", "有", "和", "就", "不", "一个", "什么", "哪些", "怎么", "如何", "通常", "一般"}


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def resolve_model(name):
    key = name.replace("/", "--")
    patterns = [os.path.join("model_cache", f"models--{key}", "snapshots", "*"),
                os.path.join("model_cache", f"models--sentence-transformers--{key}", "snapshots", "*")]
    snapshots = sorted(path for pattern in patterns for path in glob.glob(pattern) if os.path.isdir(path))
    return snapshots[-1] if snapshots else name


def tokenize(text):
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip() and token.strip() not in STOP_WORDS]


def milvus_expr(disease):
    return f"document_id == {json.dumps(disease, ensure_ascii=False)}" if disease else None


def dense_search(collection, vector, limit, nprobe, disease=None):
    hits = collection.search([vector], "dense_vector", {"metric_type": "IP", "params": {"nprobe": nprobe}},
        limit=limit, expr=milvus_expr(disease), output_fields=["document_id", "content", "metadata"])[0]
    rows = []
    for hit in hits:
        metadata = hit.entity.get("metadata") or {}
        rows.append({"id": str(hit.id), "score": float(hit.distance),
                     "document_id": hit.entity.get("document_id"), "content": hit.entity.get("content"),
                     "doc_type": metadata.get("type", "") if isinstance(metadata, dict) else ""})
    return rows


def bm25_search(connection, query, limit, disease=None):
    terms = tokenize(query)
    if not terms:
        return []
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    sql = "SELECT d.doc_id,d.disease,d.doc_type,d.content,bm25(documents_fts) AS rank FROM documents_fts JOIN documents d ON d.rowid=documents_fts.rowid WHERE documents_fts MATCH ?"
    params = [match]
    if disease:
        sql += " AND d.disease = ?"
        params.append(disease)
    # BM25 ties are common for short structured fields.  A stable ID tie-break
    # makes repeated runs reproducible and keeps RRF deterministic.
    sql += " ORDER BY rank, d.doc_id LIMIT ?"
    params.append(limit)
    return [{"id": row[0], "document_id": row[1], "doc_type": row[2], "content": row[3], "score": -float(row[4])}
            for row in connection.execute(sql, params).fetchall()]


def rrf(dense, sparse, limit, rrf_k=60):
    merged = {}
    for source, rows in (("dense", dense), ("bm25", sparse)):
        for rank, row in enumerate(rows, 1):
            item = merged.setdefault(row["id"], {**row, "score": 0.0, "sources": []})
            item["score"] += 1.0 / (rrf_k + rank)
            item["sources"].append(source)
    # A deterministic secondary key is required for reproducible experiments:
    # RRF ties are common when the two ranked lists have symmetric positions.
    return sorted(merged.values(), key=lambda row: (-row["score"], row["id"]))[:limit]


def rerank(model, query, query_vector, candidates, limit, type_boost=0.05):
    if not candidates:
        return []
    vectors = model.encode([row["content"] for row in candidates], normalize_embeddings=True,
                           show_progress_bar=False, batch_size=min(64, len(candidates)))
    query_terms = set(tokenize(query))
    type_terms = {
        "疾病症状": ("症状", "表现"), "疾病病因": ("病因", "因素", "引起"),
        "预防措施": ("预防", "避免"), "检查项目": ("检查", "化验"),
        "药品": ("药", "药物", "用药"), "宜吃食物": ("吃", "饮食"), "忌吃食物": ("别吃", "忌", "饮食"),
        "疾病简介": ("处理", "治疗", "怎么办"),
    }
    ranked = []
    for row, vector in zip(candidates, vectors):
        content_terms = set(tokenize(row["content"]))
        overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
        type_bonus = 1.0 if any(term in query for term in type_terms.get(row.get("doc_type", ""), ())) else 0.0
        item = dict(row)
        item["score"] = 0.80 * float(np.dot(query_vector, vector)) + 0.15 * overlap + type_boost * type_bonus
        ranked.append(item)
    return sorted(ranked, key=lambda row: (-row["score"], row["id"]))[:limit]


def metrics(rows):
    eligible = [row for row in rows if row["relevant_doc_ids"]]
    result = {}
    for k in KS:
        result[f"Recall@{k}"] = sum(len(set(row["retrieved"][:k]) & set(row["relevant_doc_ids"])) / len(row["relevant_doc_ids"]) for row in eligible) / len(eligible)
        result[f"HitRate@{k}"] = sum(bool(set(row["retrieved"][:k]) & set(row["relevant_doc_ids"])) for row in eligible) / len(eligible)
    for k in (5, 10):
        result[f"Precision@{k}"] = sum(len(set(row["retrieved"][:k]) & set(row["relevant_doc_ids"])) / k for row in eligible) / len(eligible)
    result["MRR"] = sum(1 / row["first_hit_rank"] if row["first_hit_rank"] else 0 for row in eligible) / len(eligible)
    result["no_hit_rate"] = sum(row["first_hit_rank"] is None for row in eligible) / len(eligible)
    latencies = [row["latency_ms"] for row in rows]
    rerank_latencies = [row.get("rerank_latency_ms", 0.0) for row in rows]
    result.update({"mean_latency_ms": statistics.mean(latencies), "p50_latency_ms": percentile(latencies, .50),
                   "p95_latency_ms": percentile(latencies, .95), "p99_latency_ms": percentile(latencies, .99),
                   "mean_rerank_latency_ms": statistics.mean(rerank_latencies)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--bm25-index")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", default="dense,bm25,hybrid,hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--use-disease-filter", action="store_true")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--split", choices=("validation", "test"))
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    cases = [json.loads(line) for line in open(args.dataset, encoding="utf-8") if line.strip()]
    if args.split:
        cases = [case for case in cases if case.get("split") == args.split]
    connections.connect(host=os.getenv("MILVUS_HOST", "localhost"), port=os.getenv("MILVUS_PORT", "19530"))
    if not utility.has_collection(args.collection):
        raise SystemExit("collection not found")
    collection = Collection(args.collection)
    collection.load()
    model = SentenceTransformer(resolve_model(args.model), cache_folder="./model_cache", local_files_only=True)
    bm25 = sqlite3.connect(args.bm25_index) if args.bm25_index else None
    if any(mode != "dense" for mode in modes) and bm25 is None:
        raise SystemExit("--bm25-index is required for BM25/hybrid modes")
    per_mode = {mode: [] for mode in modes}
    for case in cases:
        disease = case.get("disease_filter") if args.use_disease_filter else None
        query_started = time.perf_counter()
        query_vector = model.encode(case["query"], normalize_embeddings=True).tolist()
        dense_rows = dense_search(collection, query_vector, args.candidate_k, args.nprobe, disease)
        dense_ms = (time.perf_counter() - query_started) * 1000
        sparse_rows = []
        bm25_ms = 0.0
        if any(mode in ("bm25", "hybrid", "hybrid_rerank") for mode in modes):
            bm25_started = time.perf_counter()
            sparse_rows = bm25_search(bm25, case["query"], args.candidate_k, disease)
            bm25_ms = (time.perf_counter() - bm25_started) * 1000
        hybrid_rows = []
        fusion_ms = 0.0
        if any(mode in ("hybrid", "hybrid_rerank") for mode in modes):
            hybrid_started = time.perf_counter()
            hybrid_rows = rrf(dense_rows, sparse_rows, args.candidate_k, args.rrf_k)
            fusion_ms = (time.perf_counter() - hybrid_started) * 1000
        variants = {"dense": (dense_rows[:args.top_k], dense_ms, 0.0),
                    "bm25": (sparse_rows[:args.top_k], bm25_ms, 0.0),
                    "hybrid": (hybrid_rows[:args.top_k], dense_ms + bm25_ms + fusion_ms, 0.0)}
        if "hybrid_rerank" in modes:
            rerank_started = time.perf_counter()
            reranked = rerank(model, case["query"], np.asarray(query_vector), hybrid_rows, args.top_k)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            variants["hybrid_rerank"] = (reranked, dense_ms + bm25_ms + fusion_ms + rerank_ms, rerank_ms)
        for mode in modes:
            rows, latency, rerank_latency = variants[mode]
            ids = [row["id"] for row in rows]
            relevant = case["relevant_doc_ids"]
            rank = next((index + 1 for index, value in enumerate(ids) if value in set(relevant)), None)
            per_mode[mode].append({"query_id": case["query_id"], "query": case["query"],
                "relevant_doc_ids": relevant, "retrieved": ids, "scores": [row["score"] for row in rows],
                "first_hit_rank": rank, "latency_ms": latency, "rerank_latency_ms": rerank_latency,
                "error_type": None})
    output = {"run_at": datetime.now(timezone.utc).isoformat(), "collection": args.collection,
        "entity_count": collection.num_entities, "bm25_index": args.bm25_index, "model": args.model,
        "nprobe": args.nprobe, "rrf_k": args.rrf_k, "top_k": args.top_k,
        "candidate_k": args.candidate_k, "use_disease_filter": args.use_disease_filter,
        "split": args.split,
        "dataset": args.dataset, "results": {mode: {"metrics": metrics(rows), "queries": rows} for mode, rows in per_mode.items()}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as target:
        json.dump(output, target, ensure_ascii=False, indent=2)
    print(json.dumps({mode: value["metrics"] for mode, value in output["results"].items()}, ensure_ascii=False, indent=2))
    if bm25:
        bm25.close()
    connections.disconnect("default")


if __name__ == "__main__":
    main()
