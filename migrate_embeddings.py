"""Build a versioned medical QA vector collection safely and reproducibly."""
import argparse
import hashlib
import glob
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer

from vector_db.config import COLLECTIONS, EMBEDDING_CONFIG, MILVUS_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.join(BASE_DIR, "data", "medical_new_2.json")
DEFAULT_COLLECTION = os.getenv("COLLECTION_MEDICAL_QA_BUILD", f"{COLLECTIONS['medical_qa']}_v2")
BUILD_RECORD_DIR = os.path.join(BASE_DIR, "data", "build_records")
MAX_VARCHAR_BYTES = 3800
MAX_ID_BYTES = 240


def truncate_utf8(value: str, max_bytes: int) -> str:
    value = str(value or "")
    data = value.encode("utf-8")
    return value if len(data) <= max_bytes else data[:max_bytes].decode("utf-8", errors="ignore")


def stable_id(*parts: str) -> str:
    raw = "|".join(str(p or "") for p in parts)
    compact = truncate_utf8(raw.replace(" ", "_"), MAX_ID_BYTES)
    return truncate_utf8(f"{compact}_{uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:12]}", 256)


def build_medical_documents(path: str):
    docs_by_id = {}
    stats = {"source_count": 0, "invalid_source_count": 0, "duplicate_count": 0, "empty_text_count": 0}

    def add(name, doc_type, content, extra_id):
        content = truncate_utf8(content, MAX_VARCHAR_BYTES).strip()
        if not content:
            stats["empty_text_count"] += 1
            return
        disease = truncate_utf8(name, 256)
        doc = {"id": stable_id(name, doc_type, extra_id), "document_id": disease,
               "parent_id": disease, "content": content,
               "metadata": {"type": doc_type, "disease": disease}}
        if doc["id"] in docs_by_id:
            stats["duplicate_count"] += 1
        docs_by_id[doc["id"]] = doc

    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_source_count"] += 1
                continue
            stats["source_count"] += 1
            name = str(item.get("name", "")).strip()
            if not name:
                stats["empty_text_count"] += 1
                continue
            if item.get("desc"):
                add(name, "疾病简介", f"{name}：{item['desc']}", "intro")
            if item.get("cause"):
                add(name, "疾病病因", f"{name}的病因：{item['cause']}", "cause")
            if item.get("prevent"):
                add(name, "预防措施", f"{name}的预防措施：{item['prevent']}", "prevent")
            for value in item.get("symptom", []):
                add(name, "疾病症状", f"{name}的症状包括：{str(value).rstrip('...')}", value)
            for value in item.get("common_drug", []) + item.get("recommand_drug", []):
                add(name, "药品", f"{name}可以使用的药品：{value}", value)
            for value in item.get("do_eat", []) + item.get("recommand_eat", []):
                add(name, "宜吃食物", f"{name}患者宜吃：{value}", value)
            for value in item.get("not_eat", []):
                add(name, "忌吃食物", f"{name}患者忌吃：{value}", value)
            for value in item.get("check", []):
                add(name, "检查项目", f"{name}需要进行的检查：{value}", value)
    return list(docs_by_id.values()), stats


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model(model_name: str) -> str:
    cache_key = model_name.replace("/", "--")
    patterns = [
        os.path.join(BASE_DIR, "model_cache", f"models--{cache_key}", "snapshots", "*"),
        os.path.join(BASE_DIR, "model_cache", f"models--sentence-transformers--{cache_key}", "snapshots", "*"),
    ]
    paths = sorted(path for pattern in patterns for path in glob.glob(pattern) if os.path.isdir(path))
    if paths:
        return paths[-1]
    return model_name


def create_collection(name: str) -> Collection:
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_CONFIG["dim"]),
    ], description="Versioned medical QA embeddings")
    collection = Collection(name=name, schema=schema)
    collection.create_index("dense_vector", {"index_type": MILVUS_CONFIG["index_type"],
        "metric_type": MILVUS_CONFIG["metric_type"], "params": {"nlist": MILVUS_CONFIG["nlist"]}})
    return collection


def existing_ids(collection: Collection, ids):
    if not ids:
        return set()
    values = ",".join(json.dumps(value, ensure_ascii=False) for value in ids)
    rows = collection.query(expr=f"id in [{values}]", output_fields=["id"], limit=len(ids))
    return {str(row["id"]) for row in rows}


def atomic_json(path: str, value) -> None:
    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description="Build/resume a versioned medical QA collection")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-documents", type=int, default=0,
                        help="demo build limit; 0 builds the full source")
    parser.add_argument(
        "--allow-configured-collection", action="store_true",
        help="explicitly allow building the collection currently used by the app",
    )
    args = parser.parse_args()
    if (not args.dry_run and args.collection == COLLECTIONS["medical_qa"]
            and not args.allow_configured_collection):
        print("Refusing to write the configured app collection without --allow-configured-collection.")
        return 2
    if not os.path.isfile(args.source):
        print(f"Source file not found: {args.source}")
        return 2

    os.makedirs(BUILD_RECORD_DIR, exist_ok=True)
    record_path = os.path.join(BUILD_RECORD_DIR, f"{args.collection}.json")
    failure_path = os.path.join(BUILD_RECORD_DIR, f"{args.collection}.failures.jsonl")
    docs, source_stats = build_medical_documents(args.source)
    if args.max_documents > 0:
        docs = docs[:args.max_documents]
        source_stats["demo_limited"] = True
        source_stats["demo_document_limit"] = args.max_documents
    selected_model = resolve_model(EMBEDDING_CONFIG["model_name"])
    record = {
        "collection_name": args.collection, "source_data_version": os.path.basename(args.source),
        "source_data_hash": hash_file(args.source), "expected_documents": len(docs),
        "embedding_model": EMBEDDING_CONFIG["model_name"],
        "embedding_model_version": os.path.basename(selected_model),
        "embedding_dimension": MILVUS_CONFIG["dim"],
        "chunk_strategy": "one_document_per_structured_field_item",
        "distance_metric": MILVUS_CONFIG["metric_type"], "index_type": MILVUS_CONFIG["index_type"],
        "code_commit": None, "built_at": datetime.now(timezone.utc).isoformat(), **source_stats,
        "expected_count": len(docs), "processed_count": 0, "inserted_count": 0,
        "failed_count": 0, "skipped_count": 0, "invalid_vector_count": 0,
        "collection_entity_count": None, "status": "dry_run" if args.dry_run else "building",
    }
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    if os.path.isfile(record_path):
        with open(record_path, encoding="utf-8") as saved:
            previous = json.load(saved)
        for key in ("source_data_hash", "embedding_model", "embedding_dimension"):
            if previous.get("status") != "dry_run" and previous.get(key) != record.get(key):
                print(f"Resume metadata mismatch: {key}")
                return 2

    connections.connect(alias="default", host=MILVUS_CONFIG["host"], port=MILVUS_CONFIG["port"], timeout=10)
    collection = Collection(args.collection) if utility.has_collection(args.collection) else create_collection(args.collection)
    collection.load()
    local_only = os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
    model = SentenceTransformer(
        selected_model,
        cache_folder=os.path.join(BASE_DIR, "model_cache"),
        local_files_only=local_only,
    )
    failures = []
    started = time.perf_counter()
    for batch_number, start in enumerate(range(0, len(docs), args.batch_size), 1):
        batch = docs[start:start + args.batch_size]
        present = existing_ids(collection, [doc["id"] for doc in batch])
        missing = [doc for doc in batch if doc["id"] not in present]
        record["processed_count"] += len(batch)
        record["skipped_count"] += len(present)
        if missing:
            last_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    vectors = np.asarray(model.encode([doc["content"] for doc in missing],
                        normalize_embeddings=True, show_progress_bar=False, batch_size=min(64, len(missing))))
                    rows = []
                    for doc, vector in zip(missing, vectors):
                        if vector.shape != (MILVUS_CONFIG["dim"],) or not np.isfinite(vector).all():
                            record["invalid_vector_count"] += 1
                            failures.append({"id": doc["id"], "error_type": "invalid_vector"})
                        else:
                            rows.append({**doc, "dense_vector": vector.tolist()})
                    if rows:
                        collection.insert(rows)
                        record["inserted_count"] += len(rows)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < args.retries:
                        time.sleep(2 ** (attempt - 1))
            if last_error is not None:
                for doc in missing:
                    failures.append({"id": doc["id"], "error_type": type(last_error).__name__})
                record["failed_count"] += len(missing)
        if batch_number % args.checkpoint_every == 0 or record["processed_count"] == len(docs):
            record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
            atomic_json(record_path, record)
            if failures:
                with open(failure_path, "w", encoding="utf-8") as output:
                    for failure in failures:
                        output.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(json.dumps({key: record[key] for key in ("processed_count", "inserted_count", "skipped_count", "failed_count", "elapsed_seconds")}, ensure_ascii=False), flush=True)

    collection.flush()
    collection.load()
    record["collection_entity_count"] = collection.num_entities
    record["inserted_count"] = collection.num_entities
    record["failed_count"] = len(failures)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    record["status"] = "complete" if collection.num_entities == len(docs) and not failures else "incomplete"
    atomic_json(record_path, record)
    connections.disconnect("default")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
