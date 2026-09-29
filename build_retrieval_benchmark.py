"""Create a deterministic weak-label retrieval benchmark without copying source text."""
import argparse
import json
import os
import random
from collections import defaultdict

from migrate_embeddings import DEFAULT_SOURCE, build_medical_documents, hash_file

TEMPLATES = {
    "疾病症状": ("{disease}发作时身体可能出现哪些表现？", "symptom"),
    "疾病病因": ("哪些因素可能导致{disease}？", "cause"),
    "预防措施": ("平时怎样降低患{disease}的风险？", "prevention"),
    "检查项目": ("怀疑{disease}时通常要做哪些检查？", "examination"),
    "药品": ("治疗{disease}时有哪些常见药物？", "medication"),
    "宜吃食物": ("{disease}患者日常适合吃什么？", "diet_positive"),
    "忌吃食物": ("{disease}患者饮食上最好避开什么？", "diet_negative"),
    "疾病简介": ("能用通俗的话介绍一下{disease}吗？", "overview"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default="evaluation/retrieval_benchmark_v2.jsonl")
    parser.add_argument("--per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    docs, source_stats = build_medical_documents(args.source)
    grouped = defaultdict(list)
    for doc in docs:
        grouped[(doc["document_id"], doc["metadata"]["type"])].append(doc["id"])
    rng = random.Random(args.seed)
    rows = []
    for doc_type, (template, category) in TEMPLATES.items():
        diseases = sorted(disease for disease, kind in grouped if kind == doc_type)
        rng.shuffle(diseases)
        for disease in diseases[:args.per_category]:
            rows.append({"query_id": f"q{len(rows)+1:04d}", "query": template.format(disease=disease),
                "relevant_doc_ids": sorted(grouped[(disease, doc_type)]), "relevant_answers": [],
                "disease_filter": disease, "category": category, "difficulty": "medium",
                "source": "weak:source_metadata", "split": "test" if len(rows) % 4 == 0 else "validation"})
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"dataset": args.output, "source_data_hash": hash_file(args.source), "seed": args.seed,
        "case_count": len(rows), "validation_count": sum(row["split"] == "validation" for row in rows),
        "test_count": sum(row["split"] == "test" for row in rows), "label_type": "weak",
        "limitations": "Labels derive from source disease/type metadata and have not been reviewed by clinicians.", **source_stats}
    with open(args.output + ".meta.json", "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
