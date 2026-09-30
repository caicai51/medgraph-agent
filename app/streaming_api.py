"""
全功能流式后端 — FastAPI + SSE v3.0

功能：
  - /stream          — 融合 KG+Vector 的流式 RAG 问答（实时推送思考链路）
  - /stream/abort/{session_id} — 按会话中止生成
  - /sessions        — 会话 CRUD
  - /sessions/{id}/messages — 历史消息读取
  - /rag/trace/{session_id} — RAG 过程可观测
  - /health          — 健康检查

SSE 事件类型：
  event: status  → RAG 阶段状态（intent_recognition → knowledge_retrieval → generating → completed）
  event: think   → 思考步骤详情（searching / grading / rewriting / merging / reranking）
  event: token   → LLM 增量 token
  event: done    → 完成（含完整回答 + session_id + trace）
  event: error   → 错误
"""

import os
import json
import time
import uuid
import asyncio
import threading
import hashlib
import hmac
from typing import Dict, Any, Optional
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

import dashscope
from dashscope import Generation
from dotenv import load_dotenv

load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
os.environ["LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 数据层
from app.session_storage import SessionStorage
from app.cache_layer import CacheLayer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化、预热组件，关闭时清理资源"""
    # ===== Startup =====
    session_storage.connect()
    print("[streaming_api] v3.0 已启动")

    # 预热所有组件，消除首次请求的高延迟
    def _preheat():
        try:
            print("[Preheat] 开始预热组件...")
            t0 = time.time()
            retriever = get_unified_retriever()
            elapsed = (time.time() - t0) * 1000
            print(f"[Preheat] 预热完成，耗时: {elapsed:.0f}ms")
        except Exception as e:
            print(f"[Preheat] 预热失败: {e}")

    # 在后台线程中预热，不阻塞服务启动
    threading.Thread(target=_preheat, daemon=True).start()

    yield

    # ===== Shutdown =====
    print("[streaming_api] 正在关闭...")
    try:
        # 清理 VectorManager 资源（Milvus/PG 连接、BM25 状态保存）
        retriever = get_unified_retriever()
        retriever.vector_manager.close()
    except Exception as e:
        print(f"[streaming_api] 关闭清理失败: {e}")


app = FastAPI(title="RAGQnA Streaming Service", version="3.0.0", lifespan=lifespan)

# 添加 CORS 中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（开发环境）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 全局服务
# ---------------------------------------------------------------------------

session_storage = SessionStorage()
cache_layer = CacheLayer()

# 按 session_id 管理中止控制器（替代旧版全局单例）
_abort_controllers: Dict[str, "AbortController"] = {}
_abort_lock = threading.Lock()


class AbortController:
    """线程安全的中止控制器"""
    def __init__(self):
        self._aborted = False
        self._lock = threading.Lock()

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted

    def abort(self):
        with self._lock:
            self._aborted = True

    def reset(self):
        with self._lock:
            self._aborted = False


def get_abort_controller(session_id: str) -> AbortController:
    with _abort_lock:
        if session_id not in _abort_controllers:
            _abort_controllers[session_id] = AbortController()
        ctrl = _abort_controllers[session_id]
        ctrl.reset()
        return ctrl


# ---------------------------------------------------------------------------
# RAG 状态定义
# ---------------------------------------------------------------------------

class RAGStage(str, Enum):
    IDLE = "idle"
    INTENT_RECOGNITION = "intent_recognition"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    GRADING = "grading"
    PROMPT_CONSTRUCTION = "prompt_construction"
    GENERATING = "generating"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


# ---------------------------------------------------------------------------
# 延迟加载 UnifiedRetriever
# ---------------------------------------------------------------------------

_unified_retriever = None
_retriever_lock = threading.Lock()
_retriever_error = None


def resolve_user_id_from_request(
    request: Request,
    *,
    body_user_id: Optional[str] = None,
    query_user_id: Optional[str] = None,
    form_user_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve identity from a signed reverse-proxy assertion or explicit dev mode."""
    header_user_id = request.headers.get("X-Authenticated-User-Id")
    signature = request.headers.get("X-Authenticated-User-Signature")
    shared_secret = os.getenv("AUTH_PROXY_SHARED_SECRET", "")
    if header_user_id and signature and shared_secret:
        expected = hmac.new(shared_secret.encode("utf-8"), header_user_id.strip().encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(signature, expected):
            return header_user_id.strip()

    if os.getenv("ALLOW_UNTRUSTED_USER_ID", "false").lower() in {"1", "true", "yes", "on"}:
        # X-User-Id is accepted only in explicitly enabled local demo mode.
        # Production deployments must use the signed proxy headers above.
        for candidate in (
            request.headers.get("X-User-Id"), body_user_id,
            query_user_id, form_user_id,
        ):
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()

    return None


def get_unified_retriever():
    """延迟创建，避免启动时加载 NER 模型"""
    global _unified_retriever, _retriever_error
    if _unified_retriever is not None:
        return _unified_retriever
    if _retriever_error is not None:
        raise _retriever_error

    with _retriever_lock:
        if _unified_retriever is not None:
            return _unified_retriever
        if _retriever_error is not None:
            raise _retriever_error

        try:
            from vector_db.vector_manager import VectorManager

            vm = VectorManager()
            try:
                vm.connect_milvus()
            except Exception as e:
                print(f"[Retriever] Milvus 连接失败: {e}")

            try:
                vm.create_milvus_collection()
            except Exception:
                pass
            
            # 预加载集合到内存，避免查询时重复加载
            try:
                vm.preload_collections()
            except Exception as e:
                print(f"[Retriever] 集合预加载失败: {e}")

            from vector_db.unified_retriever import UnifiedRetriever

            _unified_retriever = UnifiedRetriever(
                vector_manager=vm,
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
                use_local=os.getenv("USE_LOCAL_MODEL", "true").lower() == "true",
                local_model=os.getenv("LOCAL_MODEL_NAME", "qwen2.5:7b"),
            )
            return _unified_retriever
        except Exception as e:
            _retriever_error = e
            raise


# ---------------------------------------------------------------------------
# SSE 工具
# ---------------------------------------------------------------------------

def _sse(event: str, data: Any) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False, default=str)}


def _persist_chat(session_id: str, query: str, answer: str, entities: Dict, trace: Dict):
    """持久化问答到 PostgreSQL + 更新缓存"""
    try:
        session_storage.add_message(session_id, "user", query,
            metadata={"entities": entities}, token_count=len(query))
        session_storage.add_message(session_id, "assistant", answer,
            metadata={"trace": trace, "entities": entities}, token_count=len(answer))
        cache_layer.invalidate_messages(session_id)
        cache_layer.cache_rag_trace(session_id, trace)
    except Exception as e:
        print(f"[streaming_api] 持久化失败: {e}")


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/")
async def root():
    return {
        "name": "RAGQnA 医疗问答流式服务 v3.0",
        "features": [
            "知识图谱+向量统一检索", "实时 RAG 过程可视化",
            "思考链路穿透", "按会话中止",
            "PostgreSQL 会话持久化", "Redis 缓存",
        ],
        "endpoints": {
            "stream": "/stream",
            "stream_abort": "/stream/abort/{session_id}",
            "sessions": "/sessions",
            "session_messages": "/sessions/{session_id}/messages",
            "rag_trace": "/rag/trace/{session_id}",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}


# ---------- /stream ----------

@app.post("/stream")
async def stream_chat(request: Request):
    """
    流式 RAG 问答 — 融合 KG + Vector，实时推送思考链路

    Body: {query, session_id?, use_local?, local_model?, model?}
    """
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id", "")
    user_id = resolve_user_id_from_request(request, body_user_id=body.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id"}, status_code=401)
    use_local = body.get("use_local", True)
    local_model = body.get("local_model", "qwen2.5:7b")
    model = body.get("model", "qwen-plus")

    if not query:
        return EventSourceResponse(_single_error("请输入问题"))

    if not session_id:
        session_id = session_storage.create_session(user_id) or str(uuid.uuid4())
        cache_layer.invalidate_user_sessions(user_id)
    else:
        session = session_storage.get_session(session_id)
        if not session or session.get("user_id") != user_id:
            return JSONResponse({"error": "会话不存在或无权限"}, status_code=404)

    async def event_generator():
        abort_ctrl = get_abort_controller(session_id)
        t0 = time.time()
        request_id = str(uuid.uuid4())

        def elapsed():
            return (time.time() - t0) * 1000

        # 性能记录
        perf_log = {
            "request_id": request_id,
            "query": query,
            "timestamps": {},
            "timing_details": {}
        }

        # ----- Phase 1: Intent -----
        # 预获取 retriever（不计入意图识别时间）
        retriever = get_unified_retriever()
        loop = asyncio.get_event_loop()

        intent_start = time.time()
        yield _sse("status", {"stage": "intent_recognition", "message": "正在识别查询意图...", "progress": 0.05})
        yield _sse("think", {"step": "intent", "detail": "分析用户问题，识别医学查询意图...", "elapsed_ms": elapsed()})

        if abort_ctrl.aborted:
            yield _sse("status", {"stage": "aborted", "message": "用户已中止"})
            return

        try:
            # 记录意图识别时间（仅包含实际意图识别，不含retriever初始化）
            perf_log["timestamps"]["intent_start"] = intent_start

            # 检索阶段（包含NER+意图+KG+向量检索+Rerank）
            retrieval_start = time.time()
            
            retrieve_result = await loop.run_in_executor(
                None,
                lambda: retriever.retrieve(
                    query=query, top_k=8, search_type="hybrid",
                    use_rerank=True, rerank_top_n=5, include_prompt=False, include_trace=True,
                    user_id=user_id  # 传递用户ID以检索用户上传的记忆
                )
            )
            
            retrieval_time = (time.time() - retrieval_start) * 1000
            perf_log["timing_details"]["retrieval_ms"] = round(retrieval_time, 1)

            entities = retrieve_result.get("entities", {})
            kg_results = retrieve_result.get("kg_results", [])
            vector_results = retrieve_result.get("vector_results", [])
            merged_results = retrieve_result.get("merged_results", [])

            # 从 retrieve_result 获取更详细的时间
            ret_trace = retrieve_result.get("trace", {})
            ret_trace_steps = ret_trace.get("steps", [])
            
            trace = {
                "steps": ret_trace_steps,
                "kg_count": len(kg_results), "vector_count": len(vector_results),
                "merged_count": len(merged_results), "source_breakdown": {},
                "entities": entities,  # 添加实体信息到trace
                "intent": retrieve_result.get("intent", ""),  # 添加意图信息
                "timing": {
                    "retrieval_ms": round(retrieval_time, 1),
                    "retrieval_trace": ret_trace
                }
            }

            # ----- Phase 2: 检索事件 -----
            yield _sse("status", {"stage": "knowledge_retrieval", "message": "正在检索知识库...", "progress": 0.15})

            if kg_results:
                yield _sse("think", {"step": "searching", "detail": f"查询知识图谱 → {len(kg_results)} 条结构化知识", "source": "kg", "count": len(kg_results), "elapsed_ms": elapsed()})
            if vector_results:
                yield _sse("think", {"step": "searching", "detail": f"检索向量数据库 → {len(vector_results)} 个相关文档", "source": "vector", "count": len(vector_results), "elapsed_ms": elapsed()})

            if abort_ctrl.aborted:
                yield _sse("status", {"stage": "aborted", "message": "用户已中止"})
                return

            # ----- Phase 3: Merge + Rerank -----
            yield _sse("think", {"step": "merging", "detail": f"合并知识图谱和向量检索结果，共 {len(merged_results)} 条", "elapsed_ms": elapsed()})
            yield _sse("think", {"step": "reranking", "detail": "正在重排结果...", "elapsed_ms": elapsed()})

            kg_in = sum(1 for r in merged_results if r.get("source") == "kg")
            vec_in = sum(1 for r in merged_results if r.get("source") == "vector")
            trace["source_breakdown"] = {"kg": kg_in, "vector": vec_in}

            yield _sse("think", {"step": "reranking", "detail": f"重排完成 → {kg_in} 知识图谱 + {vec_in} 文档", "elapsed_ms": elapsed()})

            if abort_ctrl.aborted:
                yield _sse("status", {"stage": "aborted", "message": "用户已中止"})
                return

            # ----- Phase 4: Prompt -----
            yield _sse("status", {"stage": "prompt_construction", "message": "正在构建提示词...", "progress": 0.3})
            prompt = retriever._build_unified_prompt(query, merged_results, entities)

            if abort_ctrl.aborted:
                yield _sse("status", {"stage": "aborted", "message": "用户已中止"})
                return

            # ----- Phase 5: 流式生成 -----
            yield _sse("status", {"stage": "generating", "message": "正在生成回答...", "progress": 0.4})
            accumulated = ""
            generation_failed = False
            generation_start = time.time()
            token_count = 0

            if use_local:
                import ollama
                try:
                    first_chunk = True
                    for chunk in ollama.chat(model=local_model, messages=[{"role": "user", "content": prompt}], stream=True):
                        if abort_ctrl.aborted:
                            yield _sse("status", {"stage": "aborted", "message": "用户已中止"})
                            return
                        # Ollama 流式输出格式：chunk["message"]["content"] 可能为空
                        token = chunk.get("message", {}).get("content", "")
                        if token:  # 只在有内容时计数
                            accumulated += token
                            token_count += 1
                            yield _sse("token", {"content": token})
                        # 检查是否完成
                        if chunk.get("done", False):
                            break
                        if first_chunk and not token:
                            # 第一个 chunk 可能只有 role 没有 content
                            first_chunk = False
                except Exception as e:
                    generation_failed = True
                    print(f"本地模型生成失败: {e}")
            else:
                try:
                    responses = Generation.call(model=model, prompt=prompt, stream=True, incremental_output=True)
                    for resp in responses:
                        if abort_ctrl.aborted:
                            yield _sse("status", {"stage": "aborted", "message": "用户已中止"})
                            return
                        if resp.status_code == 200:
                            token = getattr(getattr(resp, "output", None), "text", "")
                            if token:
                                accumulated += token
                                token_count += 1
                                yield _sse("token", {"content": token})
                        else:
                            generation_failed = True
                            yield _sse("error", {"content": f"生成错误: {getattr(resp, 'message', str(resp))}"})
                            break
                except Exception:
                    # raw HTTP fallback
                    try:
                        import requests
                        url = f"{os.environ['LLM_BASE_URL']}/services/aigc/text-generation/generation"
                        payload = {"model": model, "input": {"prompt": prompt}, "parameters": {"stream": True, "max_tokens": 2048, "temperature": 0.7}}
                        hdrs = {"Authorization": f"Bearer {dashscope.api_key}", "Content-Type": "application/json"}
                        r = requests.post(url, json=payload, headers=hdrs, stream=True, timeout=120)
                        for line in r.iter_lines(decode_unicode=True):
                            if abort_ctrl.aborted: break
                            if line and line.startswith("data:"):
                                s = line[5:].strip()
                                if s and s != "[DONE]":
                                    try:
                                        token = json.loads(s).get("output", {}).get("text", "")
                                        if token:
                                            accumulated += token
                                            token_count += 1
                                            yield _sse("token", {"content": token})
                                    except json.JSONDecodeError:
                                        continue
                        if not accumulated:
                            generation_failed = True
                    except Exception as e:
                        generation_failed = True
                        print(f"HTTP fallback 也失败: {e}")

            generation_time = (time.time() - generation_start) * 1000
            total_time = (time.time() - t0) * 1000
            
            # 记录性能
            perf_log["timing_details"]["generation_ms"] = round(generation_time, 1)
            perf_log["timing_details"]["total_ms"] = round(total_time, 1)
            perf_log["timing_details"]["token_count"] = token_count
            perf_log["timing_details"]["tokens_per_second"] = round(token_count / max(generation_time / 1000, 0.001), 1)

            # Fallback: 如果生成失败，使用检索到的知识构建回答
            if generation_failed and not accumulated:
                yield _sse("think", {"step": "fallback", "detail": "LLM生成失败，使用检索知识直接回答...", "elapsed_ms": 0})
                fallback_answer = _build_fallback_answer(merged_results, entities, query)
                if fallback_answer:
                    accumulated = fallback_answer
                    yield _sse("token", {"content": accumulated})
                else:
                    accumulated = "抱歉，当前服务暂时不可用。请稍后重试或联系管理员检查API配置。"
                    yield _sse("token", {"content": accumulated})

            # ----- Phase 6: 完成 -----
            if not abort_ctrl.aborted:
                yield _sse("status", {"stage": "completed", "message": "回答已生成", "progress": 1.0})
                
                # 更新 trace 中的性能信息
                trace["timing"]["generation_ms"] = round(generation_time, 1)
                trace["timing"]["total_ms"] = round(total_time, 1)
                trace["timing"]["token_count"] = token_count
                trace["timing"]["tokens_per_second"] = perf_log["timing_details"]["tokens_per_second"]
                
                # 打印性能日志
                intent_ms = (time.time() - intent_start) * 1000 - retrieval_time
                print(f"[Perf] request_id={request_id} | "
                      f"intent={max(intent_ms, 0):.0f}ms | "
                      f"retrieval={retrieval_time:.0f}ms | "
                      f"generation={generation_time:.0f}ms | "
                      f"total={total_time:.0f}ms | "
                      f"tokens={token_count} | "
                      f"tps={perf_log['timing_details']['tokens_per_second']:.1f}")
                
                _persist_chat(session_id, query, accumulated, entities, trace)
                yield _sse("done", {
                    "content": accumulated, 
                    "session_id": session_id, 
                    "trace": trace,
                    "performance": perf_log["timing_details"]
                })

        except Exception as e:
            yield _sse("error", {"content": f"服务异常，请稍后重试: {str(e)}"})

    return EventSourceResponse(event_generator())


def _single_error(msg: str):
    async def gen():
        yield _sse("error", {"content": msg})
    return gen()


# ---------- Abort ----------

@app.post("/stream/abort/{session_id}")
async def abort_generation(request: Request, session_id: str):
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    session = session_storage.get_session(session_id)
    if user_id is None:
        return JSONResponse({"error": "未认证用户"}, status_code=401)
    if not session or session.get("user_id") != user_id:
        return JSONResponse({"error": "会话不存在或无权限"}, status_code=404)
    with _abort_lock:
        ctrl = _abort_controllers.get(session_id)
    if ctrl:
        ctrl.abort()
        return {"status": "aborted", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# ---------- Sessions CRUD ----------

@app.get("/sessions")
async def list_sessions(request: Request):
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id"}, status_code=401)
    cached = cache_layer.get_cached_sessions(user_id)
    if cached is not None:
        return {"sessions": cached, "source": "cache"}
    sessions = session_storage.get_sessions(user_id)
    cache_layer.cache_session_list(user_id, sessions)
    return {"sessions": sessions, "source": "db"}


@app.post("/sessions")
async def create_session(request: Request):
    body = await request.json()
    user_id = resolve_user_id_from_request(request, body_user_id=body.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id"}, status_code=401)
    title = body.get("title", "新对话")
    sid = session_storage.create_session(user_id, title)
    if sid:
        cache_layer.invalidate_user_sessions(user_id)
        return {"session_id": sid, "user_id": user_id, "title": title}
    return JSONResponse({"error": "创建失败"}, status_code=500)


@app.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id"}, status_code=401)
    ok = session_storage.delete_session(session_id, user_id)
    if ok:
        cache_layer.invalidate_session(user_id, session_id)
        return {"status": "deleted", "session_id": session_id}
    return JSONResponse({"error": "删除失败或无权限"}, status_code=404)


# ---------- Messages ----------

@app.get("/sessions/{session_id}/messages")
async def get_messages(request: Request, session_id: str, limit: int = 50, offset: int = 0):
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    session = session_storage.get_session(session_id)
    if user_id is None:
        return JSONResponse({"error": "未认证用户"}, status_code=401)
    if not session or session.get("user_id") != user_id:
        return JSONResponse({"error": "会话不存在或无权限"}, status_code=404)
    if offset == 0:
        cached = cache_layer.get_cached_messages(session_id)
        if cached is not None:
            return {"messages": cached, "source": "cache"}
    messages = session_storage.get_messages(session_id, limit=limit, offset=offset)
    if offset == 0:
        cache_layer.cache_messages(session_id, messages)
    return {"messages": messages, "source": "db"}


# ---------- RAG Trace ----------

@app.get("/rag/trace/{session_id}")
async def get_rag_trace(request: Request, session_id: str):
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    session = session_storage.get_session(session_id)
    if user_id is None:
        return JSONResponse({"error": "未认证用户"}, status_code=401)
    if not session or session.get("user_id") != user_id:
        return JSONResponse({"error": "会话不存在或无权限"}, status_code=404)
    cached = cache_layer.get_cached_trace(session_id)
    if cached is not None:
        return {"trace": cached, "source": "cache"}
    messages = session_storage.get_messages(session_id, limit=10)
    for msg in reversed(messages):
        trace = msg.get("metadata", {}).get("trace")
        if trace:
            cache_layer.cache_rag_trace(session_id, trace)
            return {"trace": trace, "source": "db"}
    return {"trace": None, "message": "未找到 RAG trace"}


# ---------- Fallback Answer Builder ----------

def _build_fallback_answer(merged_results, entities, query):
    """当LLM生成失败时，使用检索结果直接构建回答"""
    if not merged_results:
        return "未检索到能够支持该问题的可靠医疗证据。请核对疾病或药品名称；涉及用药剂量时请咨询医生或药师。"
    
    parts = []
    parts.append("根据医疗知识库检索到的相关信息：\n\n")
    
    for i, result in enumerate(merged_results[:5], 1):
        content = result.get("content", "")
        if content:
            # 清理内容格式
            if isinstance(content, str) and content.startswith("【"):
                parts.append(f"{content}\n")
            else:
                parts.append(f"{i}. {content}\n")
    
    if entities:
        entity_str = "、".join([f"{k}：{v}" for k, v in entities.items()])
        parts.append(f"\n（识别到的实体：{entity_str}）")
    
    parts.append("\n\n温馨提示：以上信息仅供参考，具体诊断和治疗请咨询专业医生。")
    
    return "\n".join(parts)


# ===========================================================================
# 文件上传 API
# ===========================================================================

from fastapi import UploadFile, File, Form
from fastapi.responses import Response
from app.file_handler import (
    process_uploaded_file,
    detect_file_type,
    get_collection_for_type,
    get_file_types_info,
)


@app.get("/file-types")
async def get_file_types():
    """获取支持的文件类型列表"""
    return {"file_types": get_file_types_info()}


@app.post("/upload/record")
def upload_medical_record(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(None),
    record_type: str = Form("病历"),
):
    """
    上传医疗病历

    - file: 上传的文件
    - user_id: 用户ID
    - record_type: 病历类型（如：门诊、住院、检查报告）

    注意：使用同步 def 而非 async def，FastAPI 会自动放入线程池执行，
    避免嵌入生成/Milvus插入等同步阻塞操作卡住事件循环。
    """
    user_id = resolve_user_id_from_request(request, form_user_id=user_id)
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id", "status": "error"}, status_code=401)

    try:
        content = file.file.read()

        # 使用文件处理模块解析
        result = process_uploaded_file(
            filename=file.filename,
            content=content,
            user_id=user_id,
            file_type_hint="report",
        )
        
        if result["status"] == "error":
            return JSONResponse(
                status_code=400,
                content={"error": result["message"], "status": "error"}
            )
        
        # 获取 VectorManager 并入库
        try:
            retriever = get_unified_retriever()
            vm = retriever.vector_manager

            # 确保PostgreSQL连接已初始化
            if vm.postgres_conn is None or vm.postgres_conn.closed:
                if not vm.connect_postgres():
                    return JSONResponse(
                        status_code=500,
                        content={"error": "PostgreSQL数据库连接失败", "status": "error"}
                    )
                vm.create_postgres_tables()

            # 插入情景记忆（传入 document_id 保持与前端返回值一致）
            # 存储解析后的纯文本而非原始二进制，避免乱码
            parsed_text = ""
            for chunk in result["chunks"]:
                parsed_text += chunk.get("content", "") + "\n"
            success = vm.insert_scenario_memory(
                user_id=user_id,
                record_type=record_type,
                content=parsed_text.strip() if parsed_text else "",
                chunks=result["chunks"],
                document_id=result["document_id"],
            )
            
            if not success:
                return JSONResponse(
                    status_code=500,
                    content={"error": "存储到向量数据库失败", "status": "error"}
                )
        
        except Exception as e:
            print(f"[Upload] 向量存储失败: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"向量存储失败: {str(e)}", "status": "error"}
            )
        
        return {
            "status": "success",
            "message": f"病历上传成功，已分块 {result['total_chunks']} 条",
            "document_id": result["document_id"],
            "filename": file.filename,
            "file_type": "report",
            "record_type": record_type,
            "total_chunks": result["total_chunks"],
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"上传失败: {str(e)}", "status": "error"}
        )


@app.post("/upload/document")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(None),
    description: str = Form(""),
):
    """
    上传医学文档

    - file: 上传的文件
    - user_id: 用户ID
    - description: 文档描述（可选）

    注意：使用同步 def 而非 async def，FastAPI 会自动放入线程池执行。
    """
    user_id = resolve_user_id_from_request(request, form_user_id=user_id)
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id", "status": "error"}, status_code=401)

    try:
        content = file.file.read()

        # 使用文件处理模块解析
        result = process_uploaded_file(
            filename=file.filename,
            content=content,
            user_id=user_id,
            file_type_hint="document",
        )
        
        if result["status"] == "error":
            return JSONResponse(
                status_code=400,
                content={"error": result["message"], "status": "error"}
            )
        
        # 获取 VectorManager 并入库
        try:
            retriever = get_unified_retriever()
            vm = retriever.vector_manager

            # 确保PostgreSQL连接已初始化
            if vm.postgres_conn is None or vm.postgres_conn.closed:
                if not vm.connect_postgres():
                    return JSONResponse(
                        status_code=500,
                        content={"error": "PostgreSQL数据库连接失败", "status": "error"}
                    )
                vm.create_postgres_tables()

            # 插入语义记忆（传递 user_id 用于数据隔离）
            success = vm.insert_semantic_memory(
                document_id=result["document_id"],
                filename=file.filename,
                chunks=result["chunks"],
                user_id=user_id,
            )
            
            if not success:
                return JSONResponse(
                    status_code=500,
                    content={"error": "存储到向量数据库失败", "status": "error"}
                )
        
        except Exception as e:
            print(f"[Upload] 向量存储失败: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"向量存储失败: {str(e)}", "status": "error"}
            )
        
        return {
            "status": "success",
            "message": f"文档上传成功，已分块 {result['total_chunks']} 条",
            "document_id": result["document_id"],
            "filename": file.filename,
            "file_type": "document",
            "description": description,
            "total_chunks": result["total_chunks"],
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"上传失败: {str(e)}", "status": "error"}
        )


@app.get("/uploaded-records")
def get_uploaded_records(request: Request):
    """获取用户上传的病历列表（按 user_id 隔离）"""
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id", "records": [], "total": 0}, status_code=401)
    try:
        retriever = get_unified_retriever()
        vm = retriever.vector_manager

        # 确保 PostgreSQL 连接可用
        if vm.postgres_conn is None or vm.postgres_conn.closed:
            if not vm.connect_postgres():
                return JSONResponse(
                    status_code=503,
                    content={"error": "PostgreSQL数据库连接失败", "records": [], "total": 0}
                )

        cursor = vm.postgres_conn.cursor()
        cursor.execute(
            """
            SELECT id, record_type, content, uploaded_at
            FROM scenario_records
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            """,
            (user_id,)
        )
        records = []
        for row in cursor.fetchall():
            records.append({
                "document_id": row[0],
                "record_type": row[1],
                "content_preview": row[2][:100] if row[2] else "",
                "created_at": row[3].isoformat() if row[3] else "",
            })
        cursor.close()

        return {
            "user_id": user_id,
            "records": records,
            "total": len(records),
        }
    except Exception as e:
        print(f"[列表] 获取病历列表失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "records": [], "total": 0}
        )


@app.get("/uploaded-documents")
def get_uploaded_documents(request: Request):
    """获取用户上传的文档列表（按 user_id 隔离）"""
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id", "documents": [], "total": 0}, status_code=401)
    try:
        retriever = get_unified_retriever()
        vm = retriever.vector_manager

        # 确保 PostgreSQL 连接可用
        if vm.postgres_conn is None or vm.postgres_conn.closed:
            if not vm.connect_postgres():
                return JSONResponse(
                    status_code=503,
                    content={"error": "PostgreSQL数据库连接失败", "documents": [], "total": 0}
                )

        cursor = vm.postgres_conn.cursor()
        # 从 PostgreSQL 查询该用户的文档（user_id 列已在 create_postgres_tables 中创建）
        cursor.execute(
            """
            SELECT id, filename, file_type, leaf_chunks_count
            FROM documents
            WHERE collection_type = 'semantic_memory' AND user_id = %s
            ORDER BY filename ASC
            """,
            (user_id,)
        )
        documents = []
        for row in cursor.fetchall():
            documents.append({
                "document_id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "chunks_count": row[3],
            })
        cursor.close()

        return {
            "user_id": user_id,
            "documents": documents,
            "total": len(documents),
        }
    except Exception as e:
        print(f"[列表] 获取文档列表失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "documents": [], "total": 0}
        )


@app.delete("/uploaded-records/{document_id}")
def delete_uploaded_record(request: Request, document_id: str):
    """删除上传的病历记录（校验 user_id 所有权，同步清理 Milvus + BM25）"""
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id", "status": "error"}, status_code=401)
    try:
        retriever = get_unified_retriever()
        vm = retriever.vector_manager

        # 确保 PostgreSQL 连接可用
        if vm.postgres_conn is None or vm.postgres_conn.closed:
            if not vm.connect_postgres():
                return JSONResponse(
                    status_code=503,
                    content={"error": "PostgreSQL数据库连接失败", "status": "error"}
                )

        success = vm.delete_scenario_memory(document_id, user_id)
        if not success:
            return JSONResponse(
                status_code=403,
                content={"error": "无权删除此记录或记录不存在", "status": "error"}
            )

        return {"status": "success", "message": "病历记录已删除（含向量数据）"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"删除失败: {str(e)}", "status": "error"}
        )


@app.delete("/uploaded-documents/{document_id}")
def delete_uploaded_document(request: Request, document_id: str):
    """删除上传的文档（校验 user_id 所有权，同步清理 Milvus + BM25）"""
    user_id = resolve_user_id_from_request(request, query_user_id=request.query_params.get("user_id"))
    if user_id is None:
        return JSONResponse({"error": "未认证用户，请在请求头中提供 X-User-Id", "status": "error"}, status_code=401)
    try:
        retriever = get_unified_retriever()
        vm = retriever.vector_manager

        # 确保 PostgreSQL 连接可用
        if vm.postgres_conn is None or vm.postgres_conn.closed:
            if not vm.connect_postgres():
                return JSONResponse(
                    status_code=503,
                    content={"error": "PostgreSQL数据库连接失败", "status": "error"}
                )

        success = vm.delete_semantic_memory(document_id, user_id)
        if not success:
            return JSONResponse(
                status_code=403,
                content={"error": "无权删除此文档或文档不存在", "status": "error"}
            )

        return {"status": "success", "message": "文档已删除（含向量数据）"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"删除失败: {str(e)}", "status": "error"}
        )


# ===========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
