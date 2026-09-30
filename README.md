# MedGraph Agent：医疗知识图谱增强 RAG 系统

一个面向医疗问答场景的 Agentic RAG 项目。针对纯向量检索难以表达“疾病—症状—药品—检查”结构关系、容易产生跨疾病误召回的问题，系统将 Neo4j 知识图谱与 Milvus 混合检索结合，通过意图路由、并行召回、RRF 融合、重排和证据约束，生成可追溯回答。

> 技术演示项目，不提供真实医疗诊断或处方建议。

## 项目亮点

### 1. 多路检索编排

设计 `UnifiedRetriever` 作为检索 Agent 的编排核心：

- 规则优先、RoBERTa + BiLSTM NER 兜底，识别疾病、症状和药品实体。
- 根据意图并行调度 Neo4j 图谱检索与 Milvus/BM25 混合检索。
- 稠密、稀疏结果先进行分路 RRF，KG 与向量候选再进行跨来源 RRF。
- 接入 Jina/local rerank，并在单个服务异常时自动降级到可用链路。

在固定离线评测集上，Recall@5 从 **0.36 提升至 0.70**，MRR 从 **0.325 提升至 0.87**。

### 2. 通用问答与个人病历双链路

- 通用问答只访问公共医学图谱和公共向量集合。
- 病历问答按服务端解析的 `user_id` 检索本人数据。
- PostgreSQL、Milvus、Neo4j 和会话接口均执行用户归属校验。
- 检索降级时继续保留租户过滤，避免异常路径造成数据越权。

### 3. 双层幻觉抑制

- 检索层：疾病实体过滤、跨疾病候选降权、未知药品证据门控。
- 生成层：Prompt 强制基于检索证据回答；证据不足时明确拒答。
- 回答保留 KG/Vector 来源与 RAG trace，便于定位召回和生成问题。

### 4. 工程化与可演示性

- FastAPI + SSE 流式生成，Streamlit 展示检索、重排和回答过程。
- PostgreSQL 持久化会话，Redis 缓存并支持无缓存降级。
- 支持 PDF、DOCX、TXT、Markdown 和病历文件上传入库。
- Ollama 与 DashScope 双模型后端，可在内网使用本地模型。
- Docker Compose 编排 Milvus、Neo4j、PostgreSQL、Redis、MinIO 和 etcd。
- 提供检索指标、端到端回归、权限边界与降级路径测试。

## 系统架构

```text
User Query
    │
    ▼
NER + Intent Router
    │
    ├───────────────┐
    ▼               ▼
Neo4j KG       Milvus Dense + BM25
    │               │
    └───────┬───────┘
            ▼
      RRF Fusion
            ▼
   Jina / Local Rerank
            ▼
 Evidence Guard + Prompt
            ▼
   Ollama / DashScope LLM
            ▼
 Answer + Sources + Trace
```

## 技术栈

`Python` · `FastAPI` · `Streamlit` · `Milvus 2.3` · `Neo4j 5` · `PostgreSQL 16` · `Redis 7` · `RoBERTa + BiLSTM` · `Ollama` · `DashScope` · `Docker Compose`

## 核心代码

```text
app/streaming_api.py             FastAPI、SSE、会话和上传接口
app/webui_streaming.py           Streamlit 演示界面
vector_db/unified_retriever.py   意图路由、并行检索、RRF、重排、Prompt
vector_db/vector_manager.py      Milvus、BM25、用户过滤与降级
app/file_handler.py              文档解析、病历结构化、患者图谱写入
app/session_storage.py           PostgreSQL 会话持久化
app/cache_layer.py               Redis 缓存
scripts/                         数据导入、索引构建与离线评测脚本
evaluation/                      固定离线评测集
tests/                           指标、权限、并行编排和降级测试
```

## 快速演示

### 模型准备

仓库不提交模型权重和本地缓存。首次构建会自动下载
`paraphrase-multilingual-MiniLM-L12-v2` 作为中文嵌入模型。

规则 NER 可直接运行；如需启用完整 RoBERTa + BiLSTM NER，请准备：

```text
model/
├── best_roberta_rnn_model_ent_aug.pt
└── chinese-roberta-wwm-ext/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── vocab.txt
```

其中 RoBERTa 底座可从 Hugging Face 的 `hfl/chinese-roberta-wwm-ext` 获取；
`best_roberta_rnn_model_ent_aug.pt` 为项目微调权重，应通过 Release 或独立模型存储分发。

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

首次启动会自动创建演示图谱和向量集合。

- WebUI：<http://localhost:8501>
- API 文档：<http://localhost:8000/docs>
- Neo4j Browser：<http://localhost:7474>

生成模型可选择 Ollama 或 DashScope：

```env
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=qwen2.5:7b

# 或使用 DashScope
# USE_LOCAL_MODEL=false
# DASHSCOPE_API_KEY=your_key
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 说明

- 指标来自项目固定离线评测集，用于比较检索策略迭代效果。
- 本地演示允许使用 `X-User-Id`；生产部署应关闭 `ALLOW_UNTRUSTED_USER_ID` 并接入签名认证代理。
- 完整数据构建可将 `KG_IMPORT_LIMIT` 和 `VECTOR_IMPORT_LIMIT` 设置为 `0`。
