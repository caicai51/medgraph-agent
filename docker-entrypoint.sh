#!/bin/bash
set -e

echo "=== RAGQnA 医疗问答系统启动 ==="

echo "1. 依赖服务已通过 Docker healthcheck 就绪"

if [ "${INIT_KNOWLEDGE_GRAPH:-true}" = "true" ]; then
  echo "2. 幂等导入知识图谱数据..."
  python -m scripts.build_up_graph --website "http://neo4j:7474" --password "${NEO4J_PASSWORD}" --limit "${KG_IMPORT_LIMIT:-200}" || echo "知识图谱导入失败，将使用向量/内置知识降级..."
fi

if [ "${INIT_VECTOR_DATA:-false}" = "true" ]; then
  echo "3. 构建可恢复的 Milvus 演示集合..."
  python -m scripts.migrate_embeddings --collection "${COLLECTION_MEDICAL_QA_BUILD:-medical_qa_demo}" --max-documents "${VECTOR_IMPORT_LIMIT:-2000}" --allow-configured-collection || echo "向量初始化失败，将使用 KG/内置知识降级..."
fi

echo "4. 启动流式 API 服务..."
uvicorn app.streaming_api:app --host 0.0.0.0 --port 8000 &

echo "5. 等待 API 服务启动..."
for i in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null && break
  sleep 1
done

echo "6. 启动 WebUI..."
streamlit run app/webui_streaming.py --server.address 0.0.0.0 --server.port 8501
