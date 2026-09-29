"""
向量数据库管理器（企业级增强版）

功能：
1. 支持3个独立Collection分区存储：
   - medical_qa: 海量医疗问答向量
   - scenario_memory: 情景记忆向量（用户上传病历）
   - semantic_memory: 语义记忆向量（晦涩药品说明书）
2. 双向降级机制 - Hybrid/稀疏失败自动降级为纯稠密检索
3. Jina Rerank集成 - API级精排
4. BM25状态持久化 - 落盘到data/bm25_state.json
"""

import uuid
import os
import json
import threading
import sqlite3
import warnings

# 抑制 PyMilvus ORM 弃用警告（当前版本仍可用，3.1 才移除）
warnings.filterwarnings("ignore", message=".*ORM-style PyMilvus API.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

import psycopg2
from psycopg2 import sql
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional, Union
import numpy as np
from .config import MILVUS_CONFIG, POSTGRES_CONFIG, COLLECTIONS, EMBEDDING_CONFIG, RERANK_CONFIG, SQLITE_BM25_CONFIG
import time
import jieba
from datetime import datetime

os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HUGGINGFACE_HUB_DISABLE_SYMLINKS"] = "1"

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BM25_STATE_FILE = os.path.join(_BASE_DIR, "data", "bm25_state.json")


class VectorManager:
    def __init__(self):
        self.milvus_host = MILVUS_CONFIG["host"]
        self.milvus_port = MILVUS_CONFIG["port"]
        self.dim = MILVUS_CONFIG["dim"]
        self.metric_type = MILVUS_CONFIG["metric_type"]
        self.index_type = MILVUS_CONFIG["index_type"]
        self.nprobe = MILVUS_CONFIG["nprobe"]
        self.nlist = MILVUS_CONFIG["nlist"]

        self.collection_names = COLLECTIONS
        self.collections = {}

        self.postgres_host = POSTGRES_CONFIG["host"]
        self.postgres_port = POSTGRES_CONFIG["port"]
        self.postgres_db = POSTGRES_CONFIG["database"]
        self.postgres_user = POSTGRES_CONFIG["user"]
        self.postgres_password = POSTGRES_CONFIG["password"]

        self.jina_api_key = os.getenv("JINA_API_KEY", "")
        self.jina_rerank_endpoint = "https://api.jina.ai/v1/rerank"
        self.jina_rerank_model = RERANK_CONFIG["jina_model"]
        self.jina_rerank_timeout = RERANK_CONFIG["timeout"]

        self.embedding_model = None
        self._init_embedding_model()

        self.bm25_corpus = {}
        self.bm25_corpus_ids = {}
        self.bm25_model = {}
        self.vocabulary = {}
        self.doc_freq = {}
        self.total_docs = {}

        for col_name in self.collection_names.values():
            self.bm25_corpus[col_name] = []
            self.bm25_corpus_ids[col_name] = []
            self.bm25_model[col_name] = None
            self.vocabulary[col_name] = {}
            self.doc_freq[col_name] = {}
            self.total_docs[col_name] = 0

        self._load_bm25_state()
        self.sqlite_bm25_path = SQLITE_BM25_CONFIG["medical_index_path"]
        self._sqlite_bm25 = None

        self.milvus_connected = False
        self.postgres_conn = None

        self.last_search_mode = "hybrid"
        self.fallback_triggered = False
        
        # 嵌入缓存（LRU策略，减少重复计算）
        self._embedding_cache = {}
        self._embedding_cache_max = 1000
        self._embedding_cache_hit = 0
        self._embedding_cache_miss = 0
        
        # Milvus集合缓存（避免重复加载）
        self._collection_cache = {}
        self._collection_loaded = False
        self._collection_load_lock = threading.Lock()

    def _init_embedding_model(self):
        """初始化嵌入模型

        注意：all-MiniLM-L6-v2 是英文模型，对中文医疗文本的嵌入质量极差。
        如需中文支持，推荐替换为以下模型之一（注意维度需与 MILVUS_DIM 匹配）：

        384 维（无需重建集合）：
          - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
          - intfloat/multilingual-e5-small
        768 维（需重建集合 + 修改 MILVUS_DIM=768）：
          - shibing624/text2vec-base-chinese
          - BAAI/bge-base-zh-v1.5
        """
        try:
            from sentence_transformers import SentenceTransformer
            model_name = EMBEDDING_CONFIG["model_name"]
            self.embedding_model = SentenceTransformer(
                model_name,
                cache_folder="./model_cache"
            )
            print(f"向量化模型加载成功: {model_name}")

            # 警告：检测到英文模型用于中文场景
            ENGLISH_ONLY_MODELS = {
                "all-MiniLM-L6-v2", "all-MiniLM-L12-v2",
                "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1",
            }
            if model_name in ENGLISH_ONLY_MODELS:
                print("=" * 60)
                print("⚠️  警告：当前嵌入模型为英文模型，不支持中文！")
                print(f"   模型: {model_name}")
                print("   中文医疗文本的嵌入质量将非常差（相似度 ~0.03）。")
                print("   推荐替换为 paraphrase-multilingual-MiniLM-L12-v2")
                print("   （同维度384，无需重建Milvus集合）")
                print("=" * 60)
        except Exception as e:
            print(f"警告：无法加载SentenceTransformer模型: {e}")
            print("将使用简单的TF-IDF向量化作为回退")
            self.embedding_model = None

    def _ensure_data_dir(self):
        os.makedirs("data", exist_ok=True)

    def _load_bm25_state(self):
        try:
            if os.path.exists(BM25_STATE_FILE):
                with open(BM25_STATE_FILE, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    for col_name in self.collection_names.values():
                        col_state = state.get(col_name, {})
                        self.vocabulary[col_name] = col_state.get("vocabulary", {})
                        self.doc_freq[col_name] = col_state.get("doc_freq", {})
                        self.total_docs[col_name] = col_state.get("total_docs", 0)

                        # 加载 BM25 语料和模型
                        corpus = col_state.get("corpus", [])
                        corpus_ids = col_state.get("corpus_ids", [])
                        tokenized_corpus = col_state.get("tokenized_corpus", [])
                        if corpus and tokenized_corpus:
                            self.bm25_corpus[col_name] = corpus
                            # 如果没有对应的 corpus_ids，用空字符串占位（兼容旧状态）
                            if corpus_ids and len(corpus_ids) == len(corpus):
                                self.bm25_corpus_ids[col_name] = corpus_ids
                            else:
                                self.bm25_corpus_ids[col_name] = [""] * len(corpus)
                            self.bm25_model[col_name] = BM25Okapi(tokenized_corpus)
                            print(f"已加载BM25状态[{col_name}]: 文档数={self.total_docs[col_name]}, BM25模型已就绪")
                        else:
                            print(f"已加载BM25状态[{col_name}]: 词汇量={len(self.vocabulary[col_name])}, 文档数={self.total_docs[col_name]} (无BM25模型)")
            else:
                print("未找到BM25状态文件，将创建新索引")
        except Exception as e:
            print(f"加载BM25状态失败: {e}")

    def _save_bm25_state(self):
        try:
            self._ensure_data_dir()
            state = {}
            for col_name in self.collection_names.values():
                state[col_name] = {
                    "vocabulary": self.vocabulary[col_name],
                    "doc_freq": self.doc_freq[col_name],
                    "total_docs": self.total_docs[col_name],
                    "corpus": self.bm25_corpus[col_name],
                    "corpus_ids": self.bm25_corpus_ids[col_name],
                    "tokenized_corpus": [
                        list(jieba.cut(doc)) for doc in self.bm25_corpus[col_name]
                    ] if self.bm25_corpus[col_name] else [],
                    "updated_at": datetime.now().isoformat()
                }
            with open(BM25_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(f"BM25状态已保存: data/bm25_state.json")
        except Exception as e:
            print(f"保存BM25状态失败: {e}")

    def _update_doc_freq(self, col_name: str, tokens: List[str]):
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freq[col_name][token] = self.doc_freq[col_name].get(token, 0) + 1
        self.total_docs[col_name] += 1

    def _simple_embed(self, text: str, col_name: str) -> np.ndarray:
        tokens = list(jieba.cut(text))
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        vector = np.zeros(self.dim, dtype=np.float32)
        vocab_keys = list(self.vocabulary[col_name].keys())[:self.dim]
        for i, token in enumerate(vocab_keys):
            if token in tf:
                vector[i] = tf[token] / len(tokens)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def generate_dense_embedding(self, text: str) -> np.ndarray:
        # 检查缓存
        cache_key = text.strip()
        if cache_key in self._embedding_cache:
            self._embedding_cache_hit += 1
            return self._embedding_cache[cache_key]
        
        self._embedding_cache_miss += 1
        
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                # 存入缓存
                if len(self._embedding_cache) >= self._embedding_cache_max:
                    # LRU: 删除最早的条目
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
                self._embedding_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                print(f"SentenceTransformer编码失败，使用回退方案: {e}")
                return self._simple_embed(text, list(self.collection_names.values())[0])
        else:
            return self._simple_embed(text, list(self.collection_names.values())[0])
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total = self._embedding_cache_hit + self._embedding_cache_miss
        hit_rate = self._embedding_cache_hit / total if total > 0 else 0
        return {
            "cache_size": len(self._embedding_cache),
            "cache_hit": self._embedding_cache_hit,
            "cache_miss": self._embedding_cache_miss,
            "hit_rate": round(hit_rate, 3)
        }

    def generate_sparse_embedding(self, text: str, col_name: str) -> Dict[int, float]:
        tokens = list(jieba.cut(text))

        for token in tokens:
            if token not in self.vocabulary[col_name]:
                self.vocabulary[col_name][token] = len(self.vocabulary[col_name])
        self._update_doc_freq(col_name, tokens)

        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        sparse_vec = {}
        avg_doc_len = np.mean([len(doc) for doc in self.bm25_corpus[col_name]]) if self.bm25_corpus[col_name] else 100
        k1 = 1.5
        b = 0.75

        for token, freq in tf.items():
            if token in self.vocabulary[col_name]:
                word_id = self.vocabulary[col_name][token]
                df = self.doc_freq[col_name].get(token, 1)
                idf = np.log((self.total_docs[col_name] + 1) / (df + 1)) + 1
                weight = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len(tokens) / avg_doc_len))
                sparse_vec[word_id] = weight

        return sparse_vec

    def _local_rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5,
                      entities: Dict[str, str] = None, intent: str = None) -> List[Dict[str, Any]]:
        """
        本地 Rerank：嵌入相似度 + 关键词匹配 + 实体匹配 + 意图匹配
        优化版本：针对极端案例进行针对性增强，使用缓存提升性能
        """
        try:
            if self.embedding_model is None:
                return documents[:top_n]
            
            # 使用缓存的查询嵌入（如果可用）
            query_cache_key = f"rerank_query:{query}"
            if query_cache_key in self._embedding_cache:
                query_embedding = self._embedding_cache[query_cache_key]
                self._embedding_cache_hit += 1
            else:
                query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
                self._embedding_cache_miss += 1
                if len(self._embedding_cache) >= self._embedding_cache_max:
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
                self._embedding_cache[query_cache_key] = query_embedding
            
            # 性能优化：当文档数量 <= 15 时，直接复用检索 score 作为语义相似度
            # 避免重新对所有文档做嵌入编码（这是性能瓶颈）
            # 只有在需要更精确的精排时才重新编码
            if len(documents) <= 15 and all("score" in doc for doc in documents):
                # 复用检索 score（已归一化到0~1或接近）
                raw_scores = np.array([doc.get("score", 0.0) for doc in documents])
                # 归一化到 0~1 范围
                if raw_scores.max() > 1.0 or raw_scores.min() < 0:
                    raw_min, raw_max = raw_scores.min(), raw_scores.max()
                    if raw_max > raw_min:
                        similarities = (raw_scores - raw_min) / (raw_max - raw_min)
                    else:
                        similarities = np.ones(len(documents)) * 0.5
                else:
                    similarities = raw_scores
            else:
                # 批量编码文档（精排模式）
                doc_texts = [doc["content"] for doc in documents]
                doc_embeddings = self.embedding_model.encode(doc_texts, normalize_embeddings=True)
                similarities = np.dot(doc_embeddings, query_embedding)
            
            query_words = set(jieba.lcut(query))
            
            # 查询中的重要关键词（过滤停用词）
            stop_words = {"的", "了", "是", "在", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "什么", "哪些", "怎么", "如何", "应该"}
            important_query_words = query_words - stop_words
            
            # 提取疾病实体
            disease_entity = ""
            if entities:
                disease_entity = entities.get("疾病", "")
            
            # 解析意图类型
            intent_type = ""
            if intent:
                if "药品" in intent or "用药" in intent:
                    intent_type = "药品"
                elif "症状" in intent:
                    intent_type = "症状"
                elif "治疗" in intent:
                    intent_type = "治疗"
                elif "饮食" in intent or "宜吃" in intent or "忌吃" in intent:
                    intent_type = "饮食"
                elif "检查" in intent:
                    intent_type = "检查"
                elif "预防" in intent:
                    intent_type = "预防"

            # 患者个人病历查询意图：识别用户上传的病历块并优先返回
            is_patient_query = bool(intent and "患者病历查询" in intent)
            patient_name = ""
            if is_patient_query and entities:
                patient_name = entities.get("患者", "")

            for i, doc in enumerate(documents):
                # 语义相似度分数
                semantic_score = float(similarities[i])
                
                # 内容关键词匹配
                content = doc.get("content", "")
                content_words = set(jieba.lcut(content))
                overlap = query_words & content_words
                keyword_score = len(overlap) / max(len(query_words), 1)
                
                # 重要关键词匹配（更高权重）
                important_overlap = important_query_words & content_words
                important_keyword_score = len(important_overlap) / max(len(important_query_words), 1) if important_query_words else 0
                
                # 解析文档元数据
                metadata = doc.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                elif metadata is None:
                    metadata = {}
                
                doc_type = metadata.get("type", "") if isinstance(metadata, dict) else ""
                doc_disease = metadata.get("disease", "") if isinstance(metadata, dict) else ""
                
                # ===== 上传文档（病历/文档）的 metadata 智能解析 =====
                # 上传文档的 metadata 格式: {file_type, section, chunk_index, ...}
                # 需要从 section 字段推断类型，并从内容中提取疾病实体
                file_type = metadata.get("file_type", "") if isinstance(metadata, dict) else ""
                section = metadata.get("section", "") if isinstance(metadata, dict) else ""
                
                # 从 section 映射上传文档的类型
                if not doc_type and file_type in ("report", "document"):
                    section_type_map = {
                        "patient_info": "患者病历",
                        "diagnosis": "诊断",
                        "prescription": "处方",
                        "full_text": "患者病历",
                        "examination": "检查",
                        "indicators": "关键指标",
                        "symptoms": "症状",
                        "treatment": "治疗",
                        "treatment_plan": "治疗",
                        "medication": "药品",
                        "dosage": "药品",
                        "follow_up": "预防",
                        "lifestyle": "预防",
                        "diet": "饮食",
                        "summary": "患者病历",
                    }
                    if section in section_type_map:
                        doc_type = section_type_map[section]
                    elif file_type == "report":
                        doc_type = "患者病历"
                    elif file_type == "document":
                        doc_type = "患者病历"
                
                # Fallback: 从文档ID中提取类型信息（仅针对知识库文档）
                if not doc_type:
                    doc_id = doc.get("id", "")
                    id_parts = doc_id.split("_")
                    # 跳过 chunk_ 开头的上传文档 ID
                    if not doc_id.startswith("chunk_") and len(id_parts) >= 2:
                        type_suffix_map = {
                            # 英文映射
                            "drugs": "药品", "drug": "药品",
                            "treatment": "治疗", "treat": "治疗",
                            "symptoms": "症状", "symptom": "症状",
                            "examination": "检查", "exam": "检查",
                            "diet": "饮食", "food": "饮食",
                            "cause": "病因", "causes": "病因",
                            "overview": "疾病概览", "summary": "疾病概览",
                            "infect": "传染病", "infection": "传染病",
                            # 中文映射
                            "药品": "药品", "药物": "药品", "用药": "药品",
                            "治疗": "治疗", "治疗方法": "治疗",
                            "症状": "症状",
                            "检查": "检查", "检查项目": "检查",
                            "饮食": "饮食", "饮食建议": "饮食",
                            "病因": "病因",
                            "简介": "疾病概览",
                            "预防": "预防", "预防措施": "预防",
                            "并发": "并发疾病", "并发症": "并发疾病",
                            "宜吃": "饮食", "宜吃食物": "饮食",
                            "忌吃": "饮食", "忌吃食物": "饮食",
                            "疾病使用药品": "药品",
                            "疾病症状": "症状",
                            "疾病简介": "疾病概览",
                            "疾病治疗方法": "治疗",
                            "疾病检查项目": "检查",
                            "疾病饮食建议": "饮食",
                            "疾病病因": "病因",
                            "疾病预防措施": "预防",
                        }
                        last_part = id_parts[-1].lower()
                        if last_part in type_suffix_map:
                            doc_type = type_suffix_map[last_part]
                        for key, val in type_suffix_map.items():
                            if key in doc_id:
                                doc_type = val
                                break
                    
                    # 尝试从ID中提取疾病名（仅针对知识库文档）
                    if not doc_disease and not doc_id.startswith("chunk_") and len(id_parts) >= 2:
                        prefixes_to_skip = ["kb", "kg", "leaf", "vector", "fallback"]
                        if id_parts[0].lower() in prefixes_to_skip:
                            if len(id_parts) >= 2:
                                doc_disease = id_parts[1]
                        else:
                            doc_disease = id_parts[0]
                
                # 从内容中提取疾病名（适用于上传文档和知识库文档）
                if not doc_disease:
                    # 尝试匹配已知疾病实体
                    known_diseases = [
                        "高血压", "糖尿病", "冠心病", "高脂血症", "高血脂",
                        "感冒", "咳嗽", "哮喘", "胃炎", "胃溃疡",
                        "偏头痛", "头晕", "头痛", "失眠", "焦虑",
                        "肺炎", "支气管炎", "肺气肿", "肺结核",
                        "乙肝", "丙肝", "肝硬化", "脂肪肝",
                        "关节炎", "痛风", "骨质疏松", "骨质增生",
                        "糖尿病肾病", "糖尿病视网膜病变", "糖尿病足",
                        "高血压肾病", "高血压脑病", "高血压危象",
                    ]
                    content_lower = doc.get("content", "")
                    for disease in known_diseases:
                        if disease in content_lower:
                            doc_disease = disease
                            break
                
                # Debug: 输出metadata解析结果（仅前几个文档）
                if i < 3 and len(documents) <= 10:
                    print(f"  [Metadata Debug] Doc: {doc.get('id', '')[:30]}, "
                          f"RawMetadata: {str(metadata)[:50]}, "
                          f"ParsedType: {doc_type}, "
                          f"ParsedDisease: {doc_disease}")
                
                # 1. 疾病实体匹配（乘法模型：匹配则放大，不匹配则缩小）
                disease_match = False
                if disease_entity:
                    if doc_disease == disease_entity:
                        disease_match = True
                    elif disease_entity in content:
                        disease_match = True

                # 疾病匹配乘数：匹配×1.5，不匹配×0.2，无疾病实体×1.0
                # 不匹配从 0.5→0.3→0.2 逐步收紧，压制无关疾病（头虱/虚疟/臭汗症等）
                # 排名，提升 MRR/NDCG；无疾病实体查询保持 1.0 不惩罚
                disease_multiplier = 1.5 if disease_match else (0.2 if disease_entity else 1.0)

                # 2. 意图类型匹配（加法模型）
                intent_type_aliases = {
                    "药品": ["药品", "用药指南", "药物", "疾病使用药品", "常用药物"],
                    "症状": ["症状", "疾病症状", "临床表现"],
                    "治疗": ["治疗", "治疗方法", "治疗方案", "疾病治疗方法", "治疗的方法"],
                    "饮食": ["饮食", "饮食建议", "宜吃食物", "忌吃食物", "疾病宜吃食物", "疾病忌吃食物"],
                    "检查": ["检查", "检查项目", "疾病所需检查", "化验"],
                    "预防": ["预防", "预防措施", "防治"],
                    "疾病概览": ["疾病概览", "疾病简介", "简介"],
                    "病因": ["病因", "疾病病因", "疾病的病因"],
                    "并发疾病": ["并发疾病", "并发症", "疾病并发疾病"],
                }

                intent_match = False
                if intent_type and doc_type:
                    aliases = intent_type_aliases.get(intent_type, [intent_type])
                    if doc_type in aliases:
                        intent_match = True
                    elif intent_type == "药品" and doc_type in ["治疗", "治疗方法"]:
                        intent_match = True  # 部分匹配

                # 意图匹配加成
                intent_bonus = 0.35 if intent_match else 0.0

                # 3. 查询关键词与文档类型匹配（补充信号）
                type_bonus = 0.0
                if "症状" in query and doc_type == "症状":
                    type_bonus = 0.1
                elif "药" in query and doc_type in ["药品", "用药指南"]:
                    type_bonus = 0.1
                elif ("饮食" in query or "吃" in query) and doc_type == "饮食":
                    type_bonus = 0.1
                elif ("治疗" in query or "治" in query) and doc_type == "治疗":
                    type_bonus = 0.1
                elif "检查" in query and doc_type == "检查":
                    type_bonus = 0.1

                # 患者病历块加成（针对"患者病历查询"意图：优先返回患者病历/KG患者节点）
                patient_bonus = 0.0
                if is_patient_query:
                    is_kg_patient = doc.get("source_type") in ("patient", "patient_relationship")
                    is_patient_doc = (file_type in ("report", "document")) or is_kg_patient
                    if is_patient_doc:
                        patient_bonus += 0.6
                        # KG 患者节点属性权重最高
                        if is_kg_patient:
                            patient_bonus += 0.4
                        # patient_info 且匹配患者姓名时权重最高
                        if section == "patient_info":
                            patient_bonus += 0.4
                            if patient_name and patient_name in content:
                                patient_bonus += 0.5
                        elif section == "indicators":
                            # 关键指标块直接命中目标指标，权重较高
                            patient_bonus += 0.4
                        elif section in ("diagnosis", "examination", "prescription"):
                            patient_bonus += 0.2
                    else:
                        # 通用医学知识块在病历查询中强制压低
                        patient_bonus -= 0.8

                # 综合评分：乘法+加法混合模型
                # 基础分 = 语义相似度（0~1）+ 关键词匹配 + 意图加成 + 类型加成 + 病历加成
                base_score = semantic_score + keyword_score * 0.1 + important_keyword_score * 0.1 + intent_bonus + type_bonus + patient_bonus
                # 应用疾病乘数
                combined_score = base_score * disease_multiplier
                
                doc["rerank_score"] = combined_score
                
                # Debug信息（仅针对低分案例）
                if combined_score < 0.5 and len(documents) <= 10:
                    print(f"  [Rerank Debug] Doc: {doc.get('id', '')[:30]}, "
                          f"Score: {combined_score:.3f}, "
                          f"SemScore: {semantic_score:.3f}, "
                          f"DiseaseMult: {disease_multiplier:.1f}, "
                          f"IntentMatch: {intent_match}, "
                          f"Type: {doc_type}")
            
            ranked = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)

            # 患者病历查询：只保留病历块 / 患者节点KG结果，过滤掉混入的通用知识文档
            if is_patient_query:
                def _is_patient_record(doc):
                    # KG 患者节点结果（Patient 节点属性/关联节点）
                    if doc.get("source") == "kg" and doc.get("source_type", "").startswith("patient"):
                        return True
                    meta = doc.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}
                    if not isinstance(meta, dict):
                        return False
                    return meta.get("file_type") in ("report", "document")

                patient_records = [d for d in ranked if _is_patient_record(d)]
                if patient_records:
                    ranked = patient_records

            # 分数归一化到 0-1 区间（min-max归一化）
            if ranked:
                scores = [r.get("rerank_score", 0) for r in ranked]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                
                for doc in ranked:
                    raw_score = doc.get("rerank_score", 0)
                    if score_range > 0:
                        # 归一化到 0-1
                        doc["rerank_score_normalized"] = (raw_score - min_score) / score_range
                    else:
                        doc["rerank_score_normalized"] = 1.0 if raw_score > 0 else 0.0
                
                top_score = ranked[0].get('rerank_score', 0)
                top_normalized = ranked[0].get('rerank_score_normalized', 0)
            else:
                top_score = 0
                top_normalized = 0
            
            # 输出top-3结果（用于调试极端案例）
            if len(ranked) <= 10:
                print(f"本地Rerank完成: top_score={top_score:.4f}, normalized={top_normalized:.4f}")
                for j, r in enumerate(ranked[:3]):
                    raw = r.get('rerank_score', 0)
                    norm = r.get('rerank_score_normalized', 0)
                    print(f"  [{j+1}] {r.get('id', '')[:40]} (raw={raw:.3f}, norm={norm:.3f})")
            
            return ranked[:top_n]
        except Exception as e:
            print(f"本地Rerank失败: {e}")
            import traceback
            traceback.print_exc()
            return documents[:top_n]

    def _call_jina_rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5,
                          entities: Dict[str, str] = None, intent: str = None) -> List[Dict[str, Any]]:
        if not self.jina_api_key or self.jina_api_key == "your_jina_api_key":
            print("使用本地 Rerank (嵌入相似度)")
            return self._local_rerank(query, documents, top_n, entities, intent)

        try:
            import requests

            doc_texts = [{"text": doc["content"], "id": doc["id"]} for doc in documents]

            payload = {
                "model": self.jina_rerank_model,
                "query": query,
                "documents": doc_texts,
                "top_n": top_n,
                "return_documents": True
            }

            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                self.jina_rerank_endpoint,
                json=payload,
                headers=headers,
                timeout=self.jina_rerank_timeout
            )

            if response.status_code == 200:
                result = response.json()
                reranked = []
                for item in result.get("results", []):
                    doc_id = item["document"]["id"]
                    rerank_score = item["relevance_score"]
                    original_doc = next((d for d in documents if d["id"] == doc_id), None)
                    if original_doc:
                        original_doc["rerank_score"] = rerank_score
                        reranked.append(original_doc)

                print(f"Jina Rerank成功: {len(reranked)} 个结果")
                return reranked
            else:
                print(f"Jina Rerank API错误: {response.status_code}, 使用本地Rerank")
                return self._local_rerank(query, documents, top_n, entities, intent)

        except Exception as e:
            print(f"Jina Rerank调用失败: {e}, 使用本地Rerank")
            return self._local_rerank(query, documents, top_n, entities, intent)

    def rerank_with_jina(self, query: str, results: List[Dict[str, Any]], top_n: int = 5,
                        entities: Dict[str, str] = None, intent: str = None) -> List[Dict[str, Any]]:
        if not results:
            return []

        return self._call_jina_rerank(query, results, top_n, entities, intent)

    def connect_milvus(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias="default",
                    host=self.milvus_host,
                    port=self.milvus_port,
                    timeout=10
                )
                self.milvus_connected = True
                print("Milvus连接成功")
                return True
            except MilvusException as e:
                print(f"Milvus连接失败（第{attempt+1}次尝试）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return False

    def _check_milvus_health(self) -> bool:
        """检查 Milvus 是否可用，并确保默认连接已建立"""
        try:
            # 先尝试使用现有默认连接
            try:
                utility.list_collections(timeout=3)
                self.milvus_connected = True
                return True
            except Exception:
                pass

            # 重新建立默认连接
            try:
                connections.disconnect("default")
            except Exception:
                pass

            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port,
                timeout=5
            )
            utility.list_collections(timeout=3)
            self.milvus_connected = True
            print("[VectorManager] Milvus 重连成功")
            return True
        except Exception as e:
            self.milvus_connected = False
            return False

    def preload_collections(self):
        """预加载所有集合到内存，避免查询时重复加载"""
        if self._collection_loaded:
            return
        
        with self._collection_load_lock:
            if self._collection_loaded:
                return
            
            print("[VectorManager] 开始预加载Milvus集合...")
            start_time = time.time()
            
            for col_type, col_name in self.collection_names.items():
                try:
                    collection = self._load_collection_with_timeout(col_name, timeout=30)
                    if collection:
                        self._collection_cache[col_name] = collection
                        print(f"  [OK] [{col_name}] 已加载并缓存")
                    else:
                        print(f"  [FAIL] [{col_name}] 加载失败")
                except Exception as e:
                    print(f"  [FAIL] [{col_name}] 异常: {e}")
            
            self._collection_loaded = True
            elapsed = (time.time() - start_time) * 1000
            print(f"[VectorManager] 集合预加载完成，耗时: {elapsed:.0f}ms")
    
    def _load_collection_with_timeout(self, col_name: str, timeout: int = 30) -> Optional[Collection]:
        """带超时的集合加载，支持自动重建索引"""
        if not self._check_milvus_health():
            print(f"Milvus 不可用，跳过集合加载[{col_name}]")
            return None

        collection = None
        try:
            collection = Collection(col_name)
            
            # 检查索引是否存在
            try:
                indexes = collection.indexes
                if not indexes:
                    print(f"  [{col_name}] 索引不存在，正在重建...")
                    self._ensure_index(collection)
            except Exception as e:
                print(f"  [{col_name}] 索引检查失败: {e}，尝试重建...")
                try:
                    self._ensure_index(collection)
                except Exception as e2:
                    print(f"  [{col_name}] 索引重建失败: {e2}")
                    return None
            
            # 加载到内存
            try:
                collection.load(timeout=20)
                try:
                    num = collection.num_entities
                except:
                    num = "?"
                print(f"  [{col_name}] 加载成功，记录数: {num}")
                return collection
            except Exception as e:
                if "index not found" in str(e):
                    print(f"  [{col_name}] 索引丢失，重建后重试...")
                    try:
                        self._ensure_index(collection)
                        time.sleep(2)
                        collection.load(timeout=20)
                        try:
                            num = collection.num_entities
                        except:
                            num = "?"
                        print(f"  [{col_name}] 重试加载成功，记录数: {num}")
                        return collection
                    except Exception as e2:
                        print(f"  [{col_name}] 重试加载仍失败: {e2}")
                        return None
                else:
                    print(f"  [{col_name}] 加载失败: {e}")
                    return None
        except Exception as e:
            print(f"  [{col_name}] Collection 获取失败: {e}")
            return None
    
    def _get_or_load_collection(self, col_name: str) -> Optional[Collection]:
        """获取缓存的集合，如果未加载则加载

        性能优化：对于已缓存的集合，使用 num_entities 快速验证而非 load()，
        避免在每次检索时产生 10 秒超时等待。
        """
        if col_name in self._collection_cache:
            col = self._collection_cache[col_name]
            # 快速验证集合是否仍可用（毫秒级），不再调用 load() 浪费 10s 超时
            try:
                _ = col.num_entities
                return col
            except Exception:
                # 集合引用失效，从缓存中移除并重新加载
                del self._collection_cache[col_name]

        col = self._load_collection_with_timeout(col_name)
        if col:
            self._collection_cache[col_name] = col
        return col

    def _ensure_index(self, collection: Collection):
        """确保集合有索引"""
        try:
            # 先检查是否已有索引
            try:
                indexes = collection.indexes
                if indexes:
                    # 已有索引，无需重建
                    return
            except Exception:
                pass

            # 尝试删除旧索引（使用正确的参数名）
            try:
                collection.drop_index(field_name="dense_vector")
            except Exception:
                pass

            time.sleep(0.5)

            # 创建新索引
            index_params = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"nlist": self.nlist}
            }
            collection.create_index(
                field_name="dense_vector",
                index_params=index_params
            )
            print(f"  索引创建成功: dense_vector ({self.index_type})")
            time.sleep(1)
        except Exception as e:
            print(f"  索引创建失败: {e}")
            raise

    def _create_collection_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        return CollectionSchema(fields, description="医疗文档向量")

    def create_milvus_collections(self) -> bool:
        try:
            for col_name in self.collection_names.values():
                if utility.has_collection(col_name):
                    self.collections[col_name] = Collection(col_name)
                    print(f"Milvus集合已存在: {col_name}, 记录数: {self.collections[col_name].num_entities}")
                    continue

                schema = self._create_collection_schema()
                collection = Collection(name=col_name, schema=schema)

                dense_index_params = {
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": {"nlist": self.nlist}
                }
                collection.create_index("dense_vector", dense_index_params)

                self.collections[col_name] = collection
                print(f"Milvus集合创建成功: {col_name}")

            return True

        except MilvusException as e:
            print(f"创建Milvus集合失败: {e}")
            return False

    def connect_postgres(self) -> bool:
        try:
            self.postgres_conn = psycopg2.connect(
                host=self.postgres_host,
                port=self.postgres_port,
                dbname=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password
            )
            print("PostgreSQL连接成功")
            return True
        except psycopg2.Error as e:
            print(f"PostgreSQL连接失败: {e}")
            return False

    def create_postgres_tables(self) -> bool:
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法创建表")
                    return False

            cursor = self.postgres_conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id VARCHAR(64) PRIMARY KEY,
                    document_id VARCHAR(64) NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    collection_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS root_chunks (
                    id VARCHAR(64) PRIMARY KEY,
                    document_id VARCHAR(64) NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    collection_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(64) PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(100),
                    collection_type VARCHAR(50),
                    leaf_chunks_count INT DEFAULT 0,
                    user_id VARCHAR(64) DEFAULT 'default',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 兼容旧表：若 documents 表已存在但缺少 user_id 列，则补上
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id VARCHAR(64) DEFAULT 'default'")
            except Exception:
                pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vocabulary (
                    word VARCHAR(255) PRIMARY KEY,
                    word_id INT NOT NULL,
                    collection_type VARCHAR(50),
                    doc_count INT DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scenario_records (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64),
                    record_type VARCHAR(50),
                    content TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.postgres_conn.commit()
            cursor.close()
            print("PostgreSQL表创建成功")
            return True

        except psycopg2.Error as e:
            print(f"创建PostgreSQL表失败: {e}")
            self.postgres_conn.rollback()
            return False

    def insert_leaf_chunks(self, chunks: List[Dict[str, Any]], document_id: str,
                           collection_type: str = "medical_qa") -> bool:
        try:
            start_time = time.time()
            col_name = self.collection_names.get(collection_type, self.collection_names["medical_qa"])

            collection = Collection(col_name)
            collection.load()
            print(f"[性能] Milvus集合加载耗时: {time.time() - start_time:.3f}s")

            # 1. BM25 更新 - 添加新文档并重建模型
            bm25_start = time.time()
            new_texts = [chunk["content"] for chunk in chunks]
            new_ids = [chunk.get("id", str(uuid.uuid4())) for chunk in chunks]
            
            # 添加新文档到语料库
            for text, chunk_id in zip(new_texts, new_ids):
                self.bm25_corpus[col_name].append(text)
                self.bm25_corpus_ids[col_name].append(chunk_id)
            
            # 更新词频统计
            for text in new_texts:
                self.generate_sparse_embedding(text, col_name)
            
            # 重建 BM25 模型（BM25Okapi 不支持增量更新）
            tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_corpus[col_name]]
            self.bm25_model[col_name] = BM25Okapi(tokenized_corpus)
            
            self._save_bm25_state()
            print(f"[性能] BM25更新耗时: {time.time() - bm25_start:.3f}s")

            # 2. 批量嵌入生成
            embed_start = time.time()
            chunk_texts = [chunk["content"] for chunk in chunks]
            chunk_ids = [chunk.get("id", str(uuid.uuid4())) for chunk in chunks]
            
            # 批量生成嵌入向量
            if self.embedding_model and chunk_texts:
                # 检查缓存
                uncached_texts = []
                uncached_indices = []
                cached_vectors = {}
                
                for i, text in enumerate(chunk_texts):
                    cache_key = text.strip()
                    if cache_key in self._embedding_cache:
                        cached_vectors[i] = self._embedding_cache[cache_key]
                        self._embedding_cache_hit += 1
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                # 批量编码未缓存的文本
                if uncached_texts:
                    self._embedding_cache_miss += len(uncached_texts)
                    batch_embeddings = self.embedding_model.encode(
                        uncached_texts, 
                        show_progress_bar=False,
                        batch_size=min(32, len(uncached_texts))
                    )
                    
                    # 存入缓存
                    for idx, text, emb in zip(uncached_indices, uncached_texts, batch_embeddings):
                        cache_key = text.strip()
                        if len(self._embedding_cache) >= self._embedding_cache_max:
                            oldest_key = next(iter(self._embedding_cache))
                            del self._embedding_cache[oldest_key]
                        self._embedding_cache[cache_key] = emb
                        cached_vectors[idx] = emb
                else:
                    self._embedding_cache_miss += 0
                
                # 收集所有嵌入
                all_embeddings = [cached_vectors.get(i, self._simple_embed(chunk_texts[i], col_name)) 
                                 for i in range(len(chunk_texts))]
            else:
                self._embedding_cache_miss += len(chunk_texts)
                all_embeddings = [self._simple_embed(text, col_name) for text in chunk_texts]
            
            print(f"[性能] 批量嵌入生成耗时: {time.time() - embed_start:.3f}s")

            # 3. 构建数据并插入
            insert_start = time.time()
            data = []
            for i, chunk in enumerate(chunks):
                data.append({
                    "id": chunk_ids[i],
                    "document_id": document_id,
                    "parent_id": chunk.get("parent_id", ""),
                    "content": chunk_texts[i],
                    "metadata": chunk.get("metadata", {}),
                    "dense_vector": all_embeddings[i]
                })

            if data:
                collection.insert(data)
                collection.flush()

            collection.release()
            total_time = time.time() - start_time
            print(f"[性能] Milvus插入耗时: {time.time() - insert_start:.3f}s")
            print(f"成功插入 {len(data)} 个叶子分块到Milvus集合: {col_name} (总耗时: {total_time:.3f}s)")
            return True

        except MilvusException as e:
            print(f"插入叶子分块失败: {e}")
            return False
        except Exception as e:
            print(f"插入叶子分块异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def insert_scenario_memory(self, user_id: str, record_type: str, content: str,
                              chunks: List[Dict[str, Any]], document_id: str = None) -> bool:
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法插入情景记忆")
                    return False
                # 确保表存在
                self.create_postgres_tables()

            # 使用传入的 document_id（与 file_handler 生成的保持一致），避免删除时定位不到
            if document_id is None:
                document_id = str(uuid.uuid4())

            cursor = self.postgres_conn.cursor()
            cursor.execute("""
                INSERT INTO scenario_records (id, user_id, record_type, content)
                VALUES (%s, %s, %s, %s)
            """, (document_id, user_id, record_type, content))
            self.postgres_conn.commit()
            cursor.close()

            # 确保 chunks 的 metadata 中包含 user_id 和 document_id（用于 Milvus 过滤和删除）
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                if isinstance(chunk["metadata"], dict):
                    chunk["metadata"]["user_id"] = user_id
                    chunk["metadata"]["document_id"] = document_id

            success = self.insert_leaf_chunks(chunks, document_id, "scenario_memory")
            if success:
                print(f"成功插入情景记忆记录: user_id={user_id}, record_type={record_type}, document_id={document_id}")
            return success

        except Exception as e:
            print(f"插入情景记忆失败: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            return False

    def insert_semantic_memory(self, document_id: str, filename: str, chunks: List[Dict[str, Any]],
                               user_id: str = "default") -> bool:
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法插入语义记忆")
                    return False
                # 确保表存在
                self.create_postgres_tables()

            cursor = self.postgres_conn.cursor()

            # 确保 documents 表有 user_id 列（兼容旧表）
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id VARCHAR(64) DEFAULT 'default'")
                self.postgres_conn.commit()
            except Exception:
                self.postgres_conn.rollback()

            cursor.execute("""
                INSERT INTO documents (id, filename, file_type, collection_type, leaf_chunks_count, user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET filename = EXCLUDED.filename, user_id = EXCLUDED.user_id
            """, (document_id, filename, "semantic_memory", "semantic_memory", len(chunks), user_id))
            self.postgres_conn.commit()
            cursor.close()

            # 确保 chunks 的 metadata 中包含 user_id（用于 Milvus 过滤）
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                if isinstance(chunk["metadata"], dict):
                    chunk["metadata"]["user_id"] = user_id

            return self.insert_leaf_chunks(chunks, document_id, "semantic_memory")

        except Exception as e:
            print(f"插入语义记忆失败: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            return False

    def _cleanup_bm25_for_document(self, col_name: str, document_id: str) -> bool:
        """从内存 BM25 语料库中移除指定 document_id 的所有 chunk，并重建模型。

        注意：BM25Okapi 不支持增量删除，必须过滤后整体重建。
        """
        try:
            corpus_ids = self.bm25_corpus_ids.get(col_name, [])
            corpus = self.bm25_corpus.get(col_name, [])

            if not corpus_ids:
                return True

            # 需要先查出该 document_id 对应的所有 chunk_id
            # 通过 Milvus 查询拿 chunk_id 列表
            collection = self._get_or_load_collection(col_name)
            if collection is None:
                print(f"[BM25清理] 无法加载集合 {col_name}，跳过 BM25 清理")
                return False

            try:
                results = collection.query(
                    expr=f'document_id == "{document_id}"',
                    output_fields=["id"]
                )
                to_remove_ids = {r.get("id", "") for r in results} if results else set()
            except Exception as e:
                print(f"[BM25清理] 查询 chunk_id 失败: {e}")
                return False

            if not to_remove_ids:
                print(f"[BM25清理] document_id={document_id} 在 {col_name} 中无 chunk")
                return True

            # 按下标过滤，保留不属于该文档的条目
            keep_indices = [i for i, cid in enumerate(corpus_ids) if cid not in to_remove_ids]
            self.bm25_corpus[col_name] = [corpus[i] for i in keep_indices]
            self.bm25_corpus_ids[col_name] = [corpus_ids[i] for i in keep_indices]

            # 重建词频统计和 BM25 模型
            self.vocabulary[col_name] = {}
            self.doc_freq[col_name] = {}
            self.total_docs[col_name] = 0
            for text in self.bm25_corpus[col_name]:
                self.generate_sparse_embedding(text, col_name)

            if self.bm25_corpus[col_name]:
                tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_corpus[col_name]]
                self.bm25_model[col_name] = BM25Okapi(tokenized_corpus)
            else:
                self.bm25_model[col_name] = None

            self._save_bm25_state()
            print(f"[BM25清理] 已从 {col_name} 移除 {len(to_remove_ids)} 个 chunk（document_id={document_id}）")
            return True

        except Exception as e:
            print(f"[BM25清理] 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def delete_scenario_memory(self, document_id: str, user_id: str) -> bool:
        """删除情景记忆：先校验所有权，再清 Milvus + BM25，最后删 PG 元数据。

        顺序：先清向量、后删元数据，确保失败可重试（幂等）。
        """
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法删除情景记忆")
                    return False

            cursor = self.postgres_conn.cursor()
            # 1. 校验所有权（必须 id + user_id 同时匹配）
            cursor.execute(
                "SELECT id FROM scenario_records WHERE id = %s AND user_id = %s",
                (document_id, user_id)
            )
            if not cursor.fetchone():
                cursor.close()
                print(f"[删除] 无权删除或记录不存在: document_id={document_id}, user_id={user_id}")
                return False

            # 2. 清理 Milvus 向量
            col_name = self.collection_names.get("scenario_memory", "scenario_memory_vectors")
            try:
                collection = self._get_or_load_collection(col_name)
                if collection is not None:
                    collection.delete(expr=f'document_id == "{document_id}"')
                    collection.flush()
                    print(f"[删除] Milvus 向量已清理: document_id={document_id}")
            except Exception as e:
                print(f"[删除] Milvus 清理失败（继续删 BM25/PG）: {e}")

            # 3. 清理 BM25 内存语料
            self._cleanup_bm25_for_document(col_name, document_id)

            # 4. 删除 PG 元数据（放最后）
            cursor.execute(
                "DELETE FROM scenario_records WHERE id = %s AND user_id = %s",
                (document_id, user_id)
            )
            self.postgres_conn.commit()
            cursor.close()
            print(f"[删除] 情景记忆已彻底删除: document_id={document_id}")
            return True

        except Exception as e:
            print(f"[删除] 删除情景记忆失败: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            return False

    def delete_semantic_memory(self, document_id: str, user_id: str) -> bool:
        """删除语义记忆：先校验所有权，再清 Milvus + BM25，最后删 PG 元数据。"""
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法删除语义记忆")
                    return False

            cursor = self.postgres_conn.cursor()
            # 1. 校验所有权
            cursor.execute(
                "SELECT id FROM documents WHERE id = %s AND user_id = %s AND collection_type = 'semantic_memory'",
                (document_id, user_id)
            )
            if not cursor.fetchone():
                cursor.close()
                print(f"[删除] 无权删除或文档不存在: document_id={document_id}, user_id={user_id}")
                return False

            # 2. 清理 Milvus 向量
            col_name = self.collection_names.get("semantic_memory", "semantic_memory_vectors")
            try:
                collection = self._get_or_load_collection(col_name)
                if collection is not None:
                    collection.delete(expr=f'document_id == "{document_id}"')
                    collection.flush()
                    print(f"[删除] Milvus 向量已清理: document_id={document_id}")
            except Exception as e:
                print(f"[删除] Milvus 清理失败（继续删 BM25/PG）: {e}")

            # 3. 清理 BM25 内存语料
            self._cleanup_bm25_for_document(col_name, document_id)

            # 4. 删除 PG 元数据（放最后）
            cursor.execute(
                "DELETE FROM documents WHERE id = %s AND user_id = %s AND collection_type = 'semantic_memory'",
                (document_id, user_id)
            )
            self.postgres_conn.commit()
            cursor.close()
            print(f"[删除] 语义记忆已彻底删除: document_id={document_id}")
            return True

        except Exception as e:
            print(f"[删除] 删除语义记忆失败: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            return False

    def insert_parent_chunks(self, chunks: List[Dict[str, Any]], document_id: str,
                             collection_type: str = "medical_qa") -> bool:
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法插入父级分块")
                    return False

            cursor = self.postgres_conn.cursor()

            for chunk in chunks:
                chunk_id = chunk.get("id", str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO parent_chunks (id, document_id, content, chunk_index, collection_type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content
                """, (chunk_id, document_id, chunk["content"], chunk.get("chunk_index", 0), collection_type))

            self.postgres_conn.commit()
            cursor.close()
            print(f"成功插入 {len(chunks)} 个父级分块到PostgreSQL")
            return True

        except psycopg2.Error as e:
            print(f"插入父级分块失败: {e}")
            self.postgres_conn.rollback()
            return False

    def insert_root_chunks(self, chunks: List[Dict[str, Any]], document_id: str,
                           collection_type: str = "medical_qa") -> bool:
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法插入顶级分块")
                    return False

            cursor = self.postgres_conn.cursor()

            for chunk in chunks:
                chunk_id = chunk.get("id", str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO root_chunks (id, document_id, content, chunk_index, collection_type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content
                """, (chunk_id, document_id, chunk["content"], chunk.get("chunk_index", 0), collection_type))

            self.postgres_conn.commit()
            cursor.close()
            print(f"成功插入 {len(chunks)} 个顶级分块到PostgreSQL")
            return True

        except psycopg2.Error as e:
            print(f"插入顶级分块失败: {e}")
            self.postgres_conn.rollback()
            return False

    def insert_document_metadata(self, document_id: str, filename: str, file_type: str, leaf_count: int,
                                 collection_type: str = "medical_qa") -> bool:
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法插入文档元数据")
                    return False

            cursor = self.postgres_conn.cursor()

            cursor.execute("""
                INSERT INTO documents (id, filename, file_type, collection_type, leaf_chunks_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET filename = EXCLUDED.filename, file_type = EXCLUDED.file_type
            """, (document_id, filename, file_type, collection_type, leaf_count))

            self.postgres_conn.commit()
            cursor.close()
            print(f"成功插入文档元数据: {filename}")
            return True

        except psycopg2.Error as e:
            print(f"插入文档元数据失败: {e}")
            self.postgres_conn.rollback()
            return False

    def search_dense(self, query: str, top_k: int = 5, collection_type: str = "medical_qa",
                     disease_filter: str = None, user_filter: str = None,
                     report_only: bool = False) -> List[Dict[str, Any]]:
        """稠密向量搜索，支持疾病实体过滤和用户隔离过滤

        Args:
            query: 查询文本
            top_k: 返回数量
            collection_type: 集合类型
            disease_filter: 疾病实体名称，用于过滤检索结果（优先检索该疾病的文档）
            user_filter: 用户ID，用于过滤用户记忆集合（数据隔离）
            report_only: 患者病历查询专用，只检索 file_type==report 的病历块，禁止全库检索
        """
        try:
            col_name = self.collection_names.get(collection_type, self.collection_names["medical_qa"])

            collection = self._get_or_load_collection(col_name)
            if collection is None:
                return []

            query_vec = self.generate_dense_embedding(query)

            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": self.nprobe}
            }

            # 优化的检索逻辑：疾病匹配结果始终优先
            filtered_hits = []  # 疾病过滤命中的结果
            general_hits = []    # 无过滤搜索的补充结果

            # 构建用户隔离过滤表达式（用于 scenario_memory 和 semantic_memory 集合）
            # Milvus JSON 字段过滤语法：metadata["user_id"] == "xxx"
            user_expr = None
            if user_filter:
                user_literal = json.dumps(str(user_filter), ensure_ascii=False)
                user_expr = f'metadata["user_id"] == {user_literal}'
            # 患者病历查询：前置过滤只检索病历块(report)，避免全库检索
            if report_only:
                report_expr = 'metadata["file_type"] == "report"'
                user_expr = f'({user_expr}) and ({report_expr})' if user_expr else report_expr

            if disease_filter and not user_filter:
                # 疾病过滤（仅用于 medical_qa 集合）
                try:
                    disease_literal = json.dumps(str(disease_filter), ensure_ascii=False)
                    expr = f'document_id == {disease_literal}'
                    filter_results = collection.search(
                        data=[query_vec],
                        anns_field="dense_vector",
                        param=search_params,
                        limit=top_k * 2,
                        expr=expr,
                        output_fields=["document_id", "parent_id", "content", "metadata"]
                    )
                    if filter_results and len(filter_results[0]) > 0:
                        filtered_hits = list(filter_results[0])
                        print(f"  [Disease Filter] 疾病='{disease_filter}', 命中{len(filtered_hits)}条")
                except Exception as e:
                    print(f"  [Disease Filter] 过滤失败: {e}")
            elif user_filter:
                # 用户隔离过滤（用于 scenario_memory 和 semantic_memory 集合）
                try:
                    filter_results = collection.search(
                        data=[query_vec],
                        anns_field="dense_vector",
                        param=search_params,
                        limit=top_k * 2,
                        expr=user_expr,
                        output_fields=["document_id", "parent_id", "content", "metadata"]
                    )
                    if filter_results and len(filter_results[0]) > 0:
                        filtered_hits = list(filter_results[0])
                        print(f"  [User Filter] user='{user_filter}', 命中{len(filtered_hits)}条")
                except Exception as e:
                    print(f"  [User Filter] 过滤失败: {e}")

            # 如果疾病过滤结果不足，进行无过滤搜索补充
            if len(filtered_hits) < top_k and not user_filter:
                remaining = top_k - len(filtered_hits)
                try:
                    general_results = collection.search(
                        data=[query_vec],
                        anns_field="dense_vector",
                        param=search_params,
                        limit=remaining * 2,  # 获取更多候选
                        output_fields=["document_id", "parent_id", "content", "metadata"]
                    )
                    if general_results and len(general_results[0]) > 0:
                        # 过滤掉已经在 filtered_hits 中的文档
                        existing_ids = {hit.id for hit in filtered_hits}
                        for hit in general_results[0]:
                            if hit.id not in existing_ids:
                                # 检查 document_id 是否与疾病匹配
                                doc_disease = hit.entity.get("document_id", "")
                                if doc_disease == disease_filter:
                                    # 同一疾病，加入 filtered
                                    filtered_hits.append(hit)
                                else:
                                    # 不同疾病，作为补充
                                    general_hits.append(hit)
                            if len(filtered_hits) + len(general_hits) >= top_k:
                                break
                except Exception as e:
                    print(f"  [General Search] 补充搜索失败: {e}")

            # 合并结果：疾病匹配优先，然后是补充结果
            all_hits = filtered_hits + general_hits
            all_hits = all_hits[:top_k]  # 限制数量

            search_results = []
            for hit in all_hits:
                raw_metadata = hit.entity.get("metadata")

                # 解析metadata
                parsed_metadata = {}
                if raw_metadata:
                    if isinstance(raw_metadata, str):
                        try:
                            parsed_metadata = json.loads(raw_metadata)
                        except:
                            parsed_metadata = {}
                elif isinstance(raw_metadata, dict):
                    parsed_metadata = raw_metadata

                # Historical imports placed a few patient reports in the public
                # collection. Never expose those records through a public
                # medical query, even if their vector score is high.
                content = hit.entity.get("content") or ""
                patient_markers = ("本病历仅用于", "初步诊断：", "患者信息：", "现病史", "用药处方")
                if collection_type == "medical_qa" and (
                    parsed_metadata.get("file_type") in {"report", "document"}
                    or any(marker in content for marker in patient_markers)
                ):
                    continue

                # 判断是否疾病匹配
                doc_disease = parsed_metadata.get("disease", "") if parsed_metadata else ""
                is_disease_match = (doc_disease == disease_filter) if disease_filter else False

                search_results.append({
                    "id": hit.id,
                    "document_id": hit.entity.get("document_id"),
                    "parent_id": hit.entity.get("parent_id"),
                    "content": content,
                    "metadata": parsed_metadata,
                    "score": hit.score,
                    "dense_rank": len(search_results) + 1,
                    "sparse_rank": float('inf'),
                    "search_mode": "dense",
                    "collection_type": collection_type,
                    "is_disease_match": is_disease_match  # 标记是否疾病匹配
                })

            self.last_search_mode = "dense"
            return search_results

        except Exception as e:
            print(f"稠密向量搜索失败: {e}")
            return []

    def search_sparse(self, query: str, top_k: int = 5, collection_type: str = "medical_qa",
                      user_filter: str = None) -> List[Dict[str, Any]]:
        try:
            col_name = self.collection_names.get(collection_type, self.collection_names["medical_qa"])

            # The v2 public corpus uses a versioned SQLite FTS5 index.  It is
            # intentionally opt-in so an index from another collection cannot
            # silently be mixed into a production search.
            if collection_type == "medical_qa" and self.sqlite_bm25_path:
                return self._search_sqlite_bm25(query, top_k)

            if not self.bm25_model[col_name] or not self.bm25_corpus[col_name]:
                print(f"BM25模型未初始化[{col_name}]，返回空结果")
                return []

            query_tokens = list(jieba.cut(query))
            scores = self.bm25_model[col_name].get_scores(query_tokens)

            # 扩大候选池：取 top_k * 3 以便后续过滤后仍有足够结果
            candidate_k = min(top_k * 3, len(scores))
            top_indices = np.argsort(scores)[::-1][:candidate_k]

            collection = self._get_or_load_collection(col_name)
            if collection is None:
                return []

            corpus_ids = self.bm25_corpus_ids.get(col_name, [])
            corpus = self.bm25_corpus[col_name]

            # ===== 性能优化：批量查询 Milvus，避免 N 次独立 query =====
            # 1. 收集所有有效的 doc_id 以及有 id 的候选
            id_to_score = {}     # doc_id -> (score, idx)
            no_id_candidates = []  # 没有 doc_id 的候选

            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                doc_id = corpus_ids[idx] if idx < len(corpus_ids) else ""
                if doc_id:
                    id_to_score[doc_id] = (float(scores[idx]), idx)
                else:
                    content = corpus[idx] if idx < len(corpus) else ""
                    no_id_candidates.append((idx, float(scores[idx]), content))

            # 2. 一次批量查询所有有 id 的文档
            id_to_result = {}
            if id_to_score:
                try:
                    doc_ids = list(id_to_score.keys())
                    # Milvus expr 限制单次查询长度，超过 50 个 id 分批
                    batch_size = 50
                    for batch_start in range(0, len(doc_ids), batch_size):
                        batch_ids = doc_ids[batch_start:batch_start + batch_size]
                        id_list_str = ", ".join(f"'{did}'" for did in batch_ids)
                        expr = f"id in [{id_list_str}]"
                        batch_results = collection.query(
                            expr=expr,
                            output_fields=["id", "document_id", "parent_id", "content", "metadata"]
                        )
                        for r in batch_results:
                            id_to_result[r["id"]] = r
                except Exception as e:
                    print(f"BM25批量查询Milvus失败: {e}，回退到逐个查询")
                    # 回退：逐个查询
                    for doc_id in id_to_score:
                        try:
                            expr = f"id == '{doc_id}'"
                            results = collection.query(
                                expr=expr,
                                output_fields=["id", "document_id", "parent_id", "content", "metadata"]
                            )
                            if results:
                                id_to_result[doc_id] = results[0]
                        except Exception:
                            pass

            # 3. 按 BM25 分数排序构建结果
            search_results = []

            # 处理有 doc_id 且查到了 Milvus 结果的
            for doc_id, (score_val, idx) in id_to_score.items():
                if doc_id in id_to_result:
                    r = id_to_result[doc_id]
                    raw_metadata = r.get("metadata", {})
                    parsed_metadata = {}
                    if raw_metadata:
                        if isinstance(raw_metadata, str):
                            try:
                                parsed_metadata = json.loads(raw_metadata)
                            except Exception:
                                parsed_metadata = {}
                        elif isinstance(raw_metadata, dict):
                            parsed_metadata = raw_metadata

                    # 用户隔离过滤
                    if user_filter:
                        doc_user_id = parsed_metadata.get("user_id", "")
                        if str(doc_user_id) != str(user_filter):
                            continue

                    search_results.append({
                        "id": r["id"],
                        "document_id": r["document_id"],
                        "parent_id": r["parent_id"],
                        "content": r["content"],
                        "metadata": parsed_metadata,
                        "score": score_val,
                        "dense_rank": float('inf'),
                        "sparse_rank": len(search_results) + 1,
                        "search_mode": "sparse",
                        "collection_type": collection_type
                    })
                else:
                    # 查不到 Milvus 结果的，用 corpus 内容 fallback
                    content = corpus[idx] if idx < len(corpus) else ""
                    no_id_candidates.append((idx, score_val, content))

            # 处理没有 doc_id 的候选（fallback：直接用 corpus 内容，不查 Milvus）
            for idx, score_val, content in no_id_candidates:
                # Corpus-only rows carry no trustworthy tenant metadata and
                # therefore can never be returned from a personal collection.
                if user_filter:
                    continue
                if len(search_results) >= top_k:
                    break
                inferred_type = ""
                inferred_disease = ""
                doc_id = corpus_ids[idx] if idx < len(corpus_ids) else ""
                if doc_id:
                    doc_id_parts = doc_id.split("_")
                    type_suffix_map = {
                        "drugs": "药品", "drug": "药品",
                        "treatment": "治疗", "treat": "治疗",
                        "symptoms": "症状", "symptom": "症状",
                        "examination": "检查", "exam": "检查",
                        "diet": "饮食", "food": "饮食",
                        "cause": "病因", "causes": "病因",
                        "overview": "疾病概览", "summary": "疾病概览",
                        "infect": "传染病", "infection": "传染病",
                    }
                    for key, val in type_suffix_map.items():
                        if key in doc_id.lower():
                            inferred_type = val
                            break
                    if len(doc_id_parts) >= 2:
                        inferred_disease = doc_id_parts[0]

                search_results.append({
                    "id": doc_id,
                    "document_id": "",
                    "parent_id": "",
                    "content": content,
                    "metadata": {"type": inferred_type, "disease": inferred_disease},
                    "score": score_val,
                    "dense_rank": float('inf'),
                    "sparse_rank": len(search_results) + 1,
                    "search_mode": "sparse",
                    "collection_type": collection_type
                })

            # 截断到 top_k
            search_results = search_results[:top_k]
            self.last_search_mode = "sparse"
            return search_results

        except Exception as e:
            print(f"BM25稀疏向量搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _search_sqlite_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search a persistent FTS5 BM25 index built from the public corpus."""
        if not os.path.isfile(self.sqlite_bm25_path):
            print("配置的 SQLite BM25 索引不存在，跳过稀疏检索")
            return []
        try:
            if self._sqlite_bm25 is None:
                self._sqlite_bm25 = sqlite3.connect(self.sqlite_bm25_path, check_same_thread=False)
            tokens = [token.strip().lower() for token in jieba.lcut(query) if token.strip()]
            if not tokens:
                return []
            match_query = " ".join(tokens)
            rows = self._sqlite_bm25.execute(
                "SELECT d.doc_id, d.disease, d.doc_type, d.content, bm25(documents_fts) "
                "FROM documents_fts JOIN documents d ON d.rowid = documents_fts.rowid "
                "WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts), d.doc_id LIMIT ?",
                (match_query, top_k),
            ).fetchall()
            self.last_search_mode = "sparse_sqlite"
            return [{
                "id": doc_id, "document_id": disease, "parent_id": disease,
                "content": content, "metadata": {"disease": disease, "type": doc_type},
                "score": -float(rank), "dense_rank": float("inf"), "sparse_rank": index,
                "search_mode": "sparse_sqlite", "collection_type": "medical_qa",
            } for index, (doc_id, disease, doc_type, content, rank) in enumerate(rows, 1)]
        except sqlite3.Error as exc:
            print(f"SQLite BM25 检索失败: {type(exc).__name__}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5, rrf_k: int = 60,
                      collection_type: str = "medical_qa", disease_filter: str = None,
                      user_filter: str = None, report_only: bool = False) -> List[Dict[str, Any]]:
        self.fallback_triggered = False
        fallback_reason = None

        try:
            # 检查 BM25 是否已初始化，未初始化时跳过稀疏搜索避免无意义耗时
            col_name = self.collection_names.get(collection_type, self.collection_names["medical_qa"])
            in_memory_bm25 = bool(self.bm25_model.get(col_name)) and bool(self.bm25_corpus.get(col_name))
            sqlite_bm25 = (
                collection_type == "medical_qa"
                and bool(self.sqlite_bm25_path)
                and os.path.isfile(self.sqlite_bm25_path)
            )
            bm25_available = in_memory_bm25 or sqlite_bm25

            # Retrieve a sufficiently broad first-stage pool before fusion.  A
            # small UI Top-K must not silently limit the rerank candidate set.
            channel_top_k = max(top_k * 2, int(os.getenv("HYBRID_CHANNEL_TOP_K", "100")))
            dense_results = self.search_dense(query, channel_top_k, collection_type, disease_filter=disease_filter, user_filter=user_filter, report_only=report_only)

            sparse_results = []
            if bm25_available:
                sparse_results = self.search_sparse(query, channel_top_k, collection_type, user_filter=user_filter)
            else:
                print(f"BM25模型未初始化[{col_name}]，跳过稀疏搜索（纯稠密模式）")

            if not dense_results and sparse_results:
                self.fallback_triggered = True
                fallback_reason = "dense_failed"
                print(f"降级[{collection_type}]: 稠密向量搜索失败，使用稀疏搜索结果")
                return sparse_results[:top_k]

            if not sparse_results and dense_results:
                # 仅当 BM25 可用但返回空时才标记降级
                if bm25_available:
                    self.fallback_triggered = True
                    fallback_reason = "sparse_failed"
                    print(f"降级[{collection_type}]: 稀疏向量生成失败，使用稠密搜索结果")
                return dense_results[:top_k]

            if not dense_results and not sparse_results:
                print(f"[{collection_type}] 稠密和稀疏搜索均无结果（该集合无匹配内容）")
                return []

            fused_results = self.rrf_fusion(dense_results, sparse_results, rrf_k)
            return fused_results[:top_k]

        except Exception as e:
            print(f"Hybrid搜索失败[{collection_type}]，触发降级: {e}")
            self.fallback_triggered = True
            fallback_reason = "hybrid_failed"

            return self.search_dense(query, top_k, collection_type)

    def rrf_fusion(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        result_map = {}

        for rank, result in enumerate(dense_results, 1):
            doc_id = result["id"]
            if doc_id not in result_map:
                result_map[doc_id] = {
                    "id": result["id"],
                    "document_id": result["document_id"],
                    "parent_id": result["parent_id"],
                    "content": result["content"],
                    "metadata": result.get("metadata"),
                    "dense_rank": float('inf'),
                    "sparse_rank": float('inf'),
                    "rrf_score": 0.0,
                    "search_mode": "hybrid",
                    "collection_type": result.get("collection_type", "medical_qa")
                }
            result_map[doc_id]["dense_rank"] = rank

        for rank, result in enumerate(sparse_results, 1):
            doc_id = result["id"]
            if doc_id not in result_map:
                result_map[doc_id] = {
                    "id": result["id"],
                    "document_id": result["document_id"],
                    "parent_id": result["parent_id"],
                    "content": result["content"],
                    "metadata": result.get("metadata"),
                    "dense_rank": float('inf'),
                    "sparse_rank": float('inf'),
                    "rrf_score": 0.0,
                    "search_mode": "hybrid",
                    "collection_type": result.get("collection_type", "medical_qa")
                }
            result_map[doc_id]["sparse_rank"] = rank

        for doc_id in result_map:
            dense_rrf = 1.0 / (k + result_map[doc_id]["dense_rank"]) if result_map[doc_id]["dense_rank"] != float('inf') else 0.0
            sparse_rrf = 1.0 / (k + result_map[doc_id]["sparse_rank"]) if result_map[doc_id]["sparse_rank"] != float('inf') else 0.0
            result_map[doc_id]["rrf_score"] = dense_rrf + sparse_rrf

        fused_results = sorted(result_map.values(), key=lambda x: (-x["rrf_score"], x["id"]))
        return fused_results

    def search(self, query: str, top_k: int = 5, search_type: str = "hybrid",
               use_rerank: bool = True, rerank_top_n: int = 5,
               collection_type: str = "medical_qa", disease_filter: str = None,
               entities: Dict[str, str] = None, intent: str = None,
               user_id: str = None, report_only: bool = False) -> Dict[str, Any]:
        result = {
            "success": False,
            "results": [],
            "count": 0,
            "search_type": search_type,
            "search_mode": search_type,
            "fallback_triggered": False,
            "rerank_applied": False,
            "message": "",
            "collection_type": collection_type
        }

        try:
            # 对用户记忆集合，强制按 user_id 过滤（数据隔离）
            user_filter = None
            if collection_type in ("scenario_memory", "semantic_memory"):
                if not user_id:
                    result["message"] = f"[{collection_type}] 缺少用户身份，拒绝检索个人数据"
                    return result
                user_filter = str(user_id)

            if search_type == "dense":
                results = self.search_dense(query, top_k, collection_type, disease_filter=disease_filter, user_filter=user_filter, report_only=report_only)
            elif search_type == "sparse":
                results = self.search_sparse(query, top_k, collection_type, user_filter=user_filter)
            else:
                candidate_top_k = top_k
                if use_rerank:
                    candidate_top_k = max(top_k, int(os.getenv("HYBRID_RERANK_CANDIDATES", "20")))
                results = self.hybrid_search(query, candidate_top_k, collection_type=collection_type, disease_filter=disease_filter, user_filter=user_filter, report_only=report_only)

            result["fallback_triggered"] = self.fallback_triggered
            result["search_mode"] = self.last_search_mode

            if not results:
                result["message"] = f"[{collection_type}] 未找到相关结果"
                return result

            if use_rerank and search_type in ["dense", "hybrid"]:
                reranked = self.rerank_with_jina(query, results, rerank_top_n, entities=entities, intent=intent)
                if reranked != results[:rerank_top_n]:
                    results = reranked
                    result["rerank_applied"] = True
                    result["search_mode"] = f"{result['search_mode']}+rerank"

            result["success"] = True
            result["results"] = results
            result["count"] = len(results)
            result["message"] = f"[{collection_type}] 找到 {len(results)} 个相关结果"

            if result["fallback_triggered"]:
                result["message"] += " (触发了降级机制)"
            if result["rerank_applied"]:
                result["message"] += " (已应用Jina精排)"

            return result

        except Exception as e:
            result["message"] = f"[{collection_type}] 搜索失败: {str(e)}"
            # Preserve tenant and report filters during degradation. Dropping
            # these filters would turn a backend error into a cross-user leak.
            fallback_results = self.search_dense(
                query, top_k, collection_type,
                disease_filter=disease_filter,
                user_filter=user_filter,
                report_only=report_only,
            )
            if fallback_results:
                result["success"] = True
                result["results"] = fallback_results
                result["count"] = len(fallback_results)
                result["fallback_triggered"] = True
                result["message"] = f"[{collection_type}] 搜索失败，触发降级: {str(e)}"
            return result

    def multi_collection_search(self, query: str, top_k: int = 5, search_type: str = "hybrid",
                                use_rerank: bool = True, rerank_top_n: int = 5,
                                collection_types: Optional[List[str]] = None,
                                user_id: str = None, report_only: bool = False) -> Dict[str, Any]:
        if collection_types is None:
            collection_types = ["medical_qa", "scenario_memory", "semantic_memory"]

        all_results = []
        collection_stats = {}

        for col_type in collection_types:
            result = self.search(
                query,
                top_k=top_k,
                search_type=search_type,
                use_rerank=False,
                collection_type=col_type,
                user_id=user_id,  # 传递 user_id 用于记忆集合过滤
                report_only=report_only  # 患者病历查询：只检索病历块
            )
            collection_stats[col_type] = {
                "success": result["success"],
                "count": result["count"],
                "fallback_triggered": result["fallback_triggered"]
            }
            if result["success"] and result["results"]:
                all_results.extend(result["results"])

        final_result = {
            "success": len(all_results) > 0,
            "results": [],
            "count": len(all_results),
            "search_type": search_type,
            "collection_stats": collection_stats,
            "rerank_applied": False,
            "message": ""
        }

        if not all_results:
            final_result["message"] = "所有Collection均未找到相关结果"
            return final_result

        if use_rerank:
            reranked = self.rerank_with_jina(query, all_results, rerank_top_n)
            final_result["results"] = reranked
            final_result["rerank_applied"] = True
            final_result["message"] = f"多Collection检索: 合并 {len(all_results)} 条结果，精排后返回 {len(reranked)} 条"
        else:
            all_results.sort(key=lambda x: x.get("score", 0) or x.get("rrf_score", 0), reverse=True)
            final_result["results"] = all_results[:rerank_top_n]
            final_result["message"] = f"多Collection检索: 合并 {len(all_results)} 条结果，返回前 {rerank_top_n} 条"

        return final_result

    def hybrid_search_with_rerank(self, query: str, top_k: int = 5, rerank_top_n: int = 5,
                                  collection_type: str = "medical_qa") -> Dict[str, Any]:
        return self.search(query, top_k, search_type="hybrid", use_rerank=True,
                           rerank_top_n=rerank_top_n, collection_type=collection_type)

    def dense_search_with_rerank(self, query: str, top_k: int = 5, rerank_top_n: int = 5,
                                 collection_type: str = "medical_qa") -> Dict[str, Any]:
        return self.search(query, top_k, search_type="dense", use_rerank=True,
                           rerank_top_n=rerank_top_n, collection_type=collection_type)

    def get_parent_chunk(self, parent_id: str) -> Optional[Dict[str, Any]]:
        try:
            if self.postgres_conn is None or self.postgres_conn.closed:
                if not self.connect_postgres():
                    print("PostgreSQL连接失败，无法获取父级分块")
                    return None

            cursor = self.postgres_conn.cursor()
            cursor.execute("SELECT content FROM parent_chunks WHERE id = %s", (parent_id,))
            result = cursor.fetchone()
            cursor.close()

            if result:
                return {"content": result[0]}
            return None

        except psycopg2.Error as e:
            print(f"获取父级分块失败: {e}")
            return None

    def get_search_stats(self) -> Dict[str, Any]:
        stats = {
            "last_search_mode": self.last_search_mode,
            "fallback_triggered": self.fallback_triggered,
            "collections": {}
        }

        for col_type, col_name in self.collection_names.items():
            stats["collections"][col_type] = {
                "name": col_name,
                "vocabulary_size": len(self.vocabulary[col_name]),
                "total_docs": self.total_docs[col_name],
                "bm25_model": "initialized" if self.bm25_model[col_name] else "not initialized"
            }

        return stats

    def close(self):
        self._save_bm25_state()

        if self.milvus_connected:
            connections.disconnect("default")
            print("Milvus连接已关闭")

        if self.postgres_conn:
            self.postgres_conn.close()
            print("PostgreSQL连接已关闭")

    # ==========================================================================
    # 嵌入模型迁移
    # ==========================================================================
    # 迁移逻辑已迁移到独立的 migrate_embeddings.py 脚本，避免与
    # Milvus offset+limit<=16384 限制冲突。medical_qa 从 JSON 源文件重建，
    # scenario_memory/semantic_memory 通过 API 处理（均 < 16384 条）。
    # ==========================================================================
