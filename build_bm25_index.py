"""Build a persistent, versioned Chinese BM25 index from the vector source documents."""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time

import jieba

from migrate_embeddings import DEFAULT_SOURCE, build_medical_documents, hash_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "bm25", "medical_qa_vectors_v2.sqlite")
STOP_WORDS = {"的", "了", "是", "在", "有", "和", "就", "不", "一个", "什么", "哪些", "怎么", "如何", "通常", "一般"}


def tokenize(text: str) -> str:
    return " ".join(token.strip().lower() for token in jieba.lcut(text) if token.strip() and token.strip() not in STOP_WORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    docs, source_stats = build_medical_documents(args.source)
    connection = sqlite3.connect(args.output)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS documents (rowid INTEGER PRIMARY KEY, doc_id TEXT NOT NULL UNIQUE, disease TEXT NOT NULL, doc_type TEXT NOT NULL, content TEXT NOT NULL)")
    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(tokens, content='')")
    connection.execute("CREATE TABLE IF NOT EXISTS build_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    existing_hash = connection.execute("SELECT value FROM build_metadata WHERE key='source_data_hash'").fetchone()
    current_hash = hash_file(args.source)
    if existing_hash and existing_hash[0] != current_hash:
        raise SystemExit("Existing BM25 index belongs to another source hash; choose a new output path")
    started = time.perf_counter()
    inserted = skipped = 0
    for offset in range(0, len(docs), args.batch_size):
        for doc in docs[offset:offset + args.batch_size]:
            cursor = connection.execute("INSERT OR IGNORE INTO documents(doc_id,disease,doc_type,content) VALUES(?,?,?,?)",
                (doc["id"], doc["document_id"], doc["metadata"]["type"], doc["content"]))
            if cursor.rowcount:
                rowid = cursor.lastrowid
                connection.execute("INSERT INTO documents_fts(rowid,tokens) VALUES(?,?)", (rowid, tokenize(doc["content"])))
                inserted += 1
            else:
                skipped += 1
        connection.commit()
        print(json.dumps({"processed_count": min(offset + args.batch_size, len(docs)), "inserted_count": inserted, "skipped_count": skipped}, ensure_ascii=False), flush=True)
    count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
    fts_count = connection.execute("SELECT count(*) FROM documents_fts").fetchone()[0]
    metadata = {
        "source_data_version": os.path.basename(args.source), "source_data_hash": current_hash,
        "expected_count": len(docs), "document_count": count, "fts_count": fts_count,
        "tokenizer": "jieba", "engine": "sqlite_fts5_bm25", **source_stats,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "status": "complete" if count == len(docs) and fts_count == len(docs) else "incomplete",
    }
    for key, value in metadata.items():
        connection.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    manifest = os.path.splitext(args.output)[0] + ".json"
    with open(manifest, "w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if metadata["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
