import os
from dotenv import load_dotenv

load_dotenv()

MILVUS_CONFIG = {
    "host": os.getenv("MILVUS_HOST", "localhost"),
    "port": int(os.getenv("MILVUS_PORT", "19530")),
    "dim": int(os.getenv("MILVUS_DIM", "384")),
    "metric_type": os.getenv("MILVUS_METRIC", "IP"),
    "index_type": os.getenv("MILVUS_INDEX", "IVF_FLAT"),
    "nlist": int(os.getenv("MILVUS_NLIST", "1024")),
    "nprobe": int(os.getenv("MILVUS_NPROBE", "32")),
}

COLLECTIONS = {
    "medical_qa": os.getenv("COLLECTION_MEDICAL_QA", "medical_qa_vectors"),
    "scenario_memory": os.getenv("COLLECTION_SCENARIO_MEMORY", "scenario_memory_vectors"),
    "semantic_memory": os.getenv("COLLECTION_SEMANTIC_MEMORY", "semantic_memory_vectors"),
}

POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "rag_medical"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "password"),
}

CHUNK_CONFIG = {
    "leaf_chunk_size": int(os.getenv("LEAF_CHUNK_SIZE", "256")),
    "leaf_overlap": int(os.getenv("LEAF_OVERLAP", "64")),
    "parent_chunk_size": int(os.getenv("PARENT_CHUNK_SIZE", "1024")),
    "parent_overlap": int(os.getenv("PARENT_OVERLAP", "256")),
    "root_chunk_size": int(os.getenv("ROOT_CHUNK_SIZE", "4096")),
    "root_overlap": int(os.getenv("ROOT_OVERLAP", "512")),
}

EMBEDDING_CONFIG = {
    # 推荐使用支持中文的多语言模型（384维，无需重建Milvus集合）：
    #   "paraphrase-multilingual-MiniLM-L12-v2"  — 支持50+语言，含中文
    #   "intfloat/multilingual-e5-small"          — 多语言E5，384维
    # 如果希望更好效果且可重建集合，使用768维中文模型：
    #   "shibing624/text2vec-base-chinese"        — 中文语义匹配专用
    #   "BAAI/bge-base-zh-v1.5"                  — BGE中文模型
    # ⚠️ all-MiniLM-L6-v2 是英文模型，中文嵌入质量极差！
    "model_name": os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
    "max_seq_length": int(os.getenv("EMBEDDING_MAX_LENGTH", "256")),
}

RERANK_CONFIG = {
    # Jina official multilingual rerank models are a better fit for Chinese medical text
    # than the old English-only default used by this project.
    "jina_model": os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual"),
    "timeout": int(os.getenv("JINA_RERANK_TIMEOUT", "10")),
}

# Optional persistent FTS5 BM25 index for the public medical collection.  It
# must be built from the same source snapshot as COLLECTION_MEDICAL_QA.
SQLITE_BM25_CONFIG = {
    "medical_index_path": os.getenv("MEDICAL_BM25_SQLITE_PATH", ""),
}
