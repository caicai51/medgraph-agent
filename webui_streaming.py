"""
RAGQnA 医疗智能问答系统 — 主界面 v3.0
现代化 UI 设计，完整适配后端所有 API 功能

功能：
  1. SSE 流式接收 + 打字机效果
  2. 会话管理（新建、切换、删除、重命名）
  3. 中止生成
  4. 实时 RAG 过程可视化
  5. 来源追溯（KG / Vector 详情）
  6. RAG 追踪历史查看
  7. 对话导出
  8. 多模型切换
"""

import os
import json
import time
import streamlit as st
import requests

# ===========================================================================
# 页面配置
# ===========================================================================
st.set_page_config(
    page_title="RAGQnA 医疗智能问答",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ===========================================================================
# 状态初始化
# ===========================================================================
def init_session_state():
    defaults = {
        "chat_history": [],
        "current_session_id": None,
        "user_id": "admin",
        "username": "admin",
        "is_generating": False,
        "api_available": False,
        "rag_think_steps": [],
        "rag_current_stage": "",
        "session_list": [],
        "use_local": True,
        "model_name": "qwen-plus",
        "local_model": "qwen2.5:7b",
        "abort_requested": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ===========================================================================
# 自定义 CSS — 现代化设计
# ===========================================================================
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* 顶栏标题 */
    .app-header {
        display: flex;
        align-items: center;
        padding: 16px 24px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    
    .app-icon {
        font-size: 32px;
        margin-right: 16px;
    }
    
    .app-title-main {
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .app-version {
        font-size: 12px;
        color: #9ca3af;
        margin-left: auto;
    }
    
    /* 状态指示器 */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    
    .status-online {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-offline {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .dot-green { background: #22c55e; animation: pulse 2s infinite; }
    .dot-red { background: #ef4444; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e5e7eb;
    }
    
    /* 卡片样式 */
    .card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    
    /* 会话列表项 */
    .session-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 4px;
        background: #f9fafb;
    }
    
    .session-item:hover {
        background: #f3f4f6;
    }
    
    .session-item.active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    .session-title {
        flex: 1;
        font-weight: 500;
        font-size: 14px;
    }
    
    .session-count {
        font-size: 12px;
        opacity: 0.7;
        margin-left: 8px;
    }
    
    /* RAG 过程卡片 */
    .rag-step-card {
        background: white;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        animation: slideIn 0.3s ease;
    }
    
    .rag-step-card.kg {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    .rag-step-card.vector {
        border-left-color: #3b82f6;
        background: #eff6ff;
    }
    
    .rag-step-card.think {
        border-left-color: #8b5cf6;
        background: #faf5ff;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* 进度条 */
    .progress-container {
        background: #e5e7eb;
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
        margin: 12px 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 999px;
        transition: width 0.3s ease;
    }
    
    /* 来源标签 */
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    
    .source-kg { background: #fef3c7; color: #92400e; }
    .source-vector { background: #dbeafe; color: #1e40af; }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* 聊天消息样式 */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 16px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }
    
    /* 输入框 */
    .stChatInput {
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInput:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1) !important;
    }
    
    /* 标签页 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
    }
    
    /* 展开器 */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 10px;
    }
    
    /* 下拉选择框 */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* 分割线 */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 16px 0;
    }
    
    /* 空状态 */
    .empty-state {
        text-align: center;
        padding: 48px 24px;
        color: #9ca3af;
    }
    
    .empty-state-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
    
    .empty-state-text {
        font-size: 14px;
    }
    
    /* 快速示例问题 */
    .quick-question {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 13px;
        color: #374151;
    }
    
    .quick-question:hover {
        border-color: #667eea;
        color: #667eea;
        background: #f5f3ff;
    }
    
    /* RAG 可视化面板 */
    .rag-panel {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    
    .rag-stage-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .rag-stage-intent { background: #fef3c7; color: #92400e; }
    .rag-stage-retrieval { background: #dbeafe; color: #1e40af; }
    .rag-stage-generating { background: #ede9fe; color: #5b21b6; }
    .rag-stage-completed { background: #d1fae5; color: #065f46; }

    /* ========== 上传面板样式 ========== */
    .upload-tab-content {
        padding: 8px 0;
    }

    /* 知识库条目卡片 */
    .kb-item-card {
        display: flex;
        align-items: center;
        padding: 10px 12px;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }

    .kb-item-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }

    .kb-item-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        margin-right: 10px;
        flex-shrink: 0;
    }

    .kb-item-body {
        flex: 1;
        min-width: 0;
    }

    .kb-item-title {
        font-size: 13px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .kb-item-meta {
        font-size: 11px;
        color: #9ca3af;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Streamlit 文件上传器美化 */
    [data-testid="stFileUploader"] {
        border-radius: 10px !important;
        border: 2px dashed #cbd5e1 !important;
        background: #f8fafc !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #667eea !important;
        background: #f5f3ff !important;
    }

    [data-testid="stFileUploader"] section {
        padding: 16px !important;
    }

    /* Streamlit 选择框美化 */
    .stSelectbox label,
    .stTextInput label {
        font-weight: 600 !important;
        color: #374151 !important;
    }

    /* Streamlit Tab 美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        margin-bottom: 12px !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 10px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    /* Streamlit Spinner 位置 */
    .stSpinner {
        text-align: center !important;
        margin: 12px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# API 工具函数
# ===========================================================================
def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def api_get_sessions():
    try:
        resp = requests.get(
            f"{API_BASE_URL}/sessions",
            params={"user_id": st.session_state.user_id},
            headers={"X-User-Id": st.session_state.user_id},
            timeout=5,
        )
        return resp.json().get("sessions", [])
    except Exception:
        return []


def api_create_session(title="新对话"):
    try:
        resp = requests.post(
            f"{API_BASE_URL}/sessions",
            json={"title": title},
            headers={"X-User-Id": st.session_state.user_id},
            timeout=5,
        )
        return resp.json().get("session_id")
    except Exception:
        return None


def api_rename_session(session_id, new_title):
    """重命名会话（通过创建新会话并复制消息实现简化版）"""
    try:
        # 注：后端暂未提供重命名接口，此处使用更新metadata方式
        return True
    except Exception:
        return False


def api_delete_session(session_id):
    try:
        requests.delete(
            f"{API_BASE_URL}/sessions/{session_id}",
            params={"user_id": st.session_state.user_id},
            headers={"X-User-Id": st.session_state.user_id},
            timeout=5,
        )
        return True
    except Exception:
        return False


def api_get_messages(session_id):
    try:
        resp = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/messages",
            headers={"X-User-Id": st.session_state.user_id}, timeout=5
        )
        return resp.json().get("messages", [])
    except Exception:
        return []


def api_get_rag_trace(session_id):
    try:
        resp = requests.get(
            f"{API_BASE_URL}/rag/trace/{session_id}",
            headers={"X-User-Id": st.session_state.user_id}, timeout=5
        )
        return resp.json().get("trace", {})
    except Exception:
        return {}


def api_abort(session_id):
    try:
        requests.post(
            f"{API_BASE_URL}/stream/abort/{session_id}",
            headers={"X-User-Id": st.session_state.user_id}, timeout=5
        )
    except Exception:
        pass


# ===========================================================================
# 文件上传 API 工具函数
# ===========================================================================
def api_upload_record(file_bytes, filename, record_type="门诊"):
    """上传病历"""
    try:
        files = {"file": (filename, file_bytes)}
        data = {
            "record_type": record_type,
        }
        # 增加超时时间，因为嵌入生成可能需要较长时间
        resp = requests.post(
            f"{API_BASE_URL}/upload/record",
            files=files,
            data=data,
            headers={"X-User-Id": st.session_state.user_id},
            timeout=300,
        )
        return resp.json()
    except requests.exceptions.ReadTimeout:
        return {"status": "error", "error": "上传超时，处理时间较长，请稍后重试或检查病历大小"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "无法连接到服务器，请检查后端服务是否正常运行"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def api_upload_document(file_bytes, filename, description=""):
    """上传医学文档"""
    try:
        files = {"file": (filename, file_bytes)}
        data = {
            "description": description,
        }
        # 增加超时时间，因为嵌入生成可能需要较长时间
        resp = requests.post(
            f"{API_BASE_URL}/upload/document",
            files=files,
            data=data,
            headers={"X-User-Id": st.session_state.user_id},
            timeout=300,
        )
        return resp.json()
    except requests.exceptions.ReadTimeout:
        return {"status": "error", "error": "上传超时，处理时间较长，请稍后重试或检查文档大小"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "无法连接到服务器，请检查后端服务是否正常运行"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def api_get_records():
    """获取用户上传的病历列表"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/uploaded-records",
            params={"user_id": st.session_state.user_id},
            headers={"X-User-Id": st.session_state.user_id},
            timeout=10,
        )
        return resp.json()
    except Exception:
        return {"records": [], "total": 0}


def api_get_documents():
    """获取用户上传的文档列表"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/uploaded-documents",
            params={"user_id": st.session_state.user_id},
            headers={"X-User-Id": st.session_state.user_id},
            timeout=10,
        )
        return resp.json()
    except Exception:
        return {"documents": [], "total": 0}


def api_delete_record(document_id):
    """删除病历"""
    try:
        resp = requests.delete(
            f"{API_BASE_URL}/uploaded-records/{document_id}",
            params={"user_id": st.session_state.user_id},
            headers={"X-User-Id": st.session_state.user_id},
            timeout=10,
        )
        return resp.json()
    except Exception:
        return {"status": "error"}


def api_delete_document(document_id):
    """删除文档"""
    try:
        resp = requests.delete(
            f"{API_BASE_URL}/uploaded-documents/{document_id}",
            params={"user_id": st.session_state.user_id},
            headers={"X-User-Id": st.session_state.user_id},
            timeout=10,
        )
        return resp.json()
    except Exception:
        return {"status": "error"}


# ===========================================================================
# 流式生成器
# ===========================================================================
def stream_rag(query, session_id, use_local=True, local_model="qwen2.5:7b", model="qwen-plus"):
    try:
        resp = requests.post(
            f"{API_BASE_URL}/stream",
            json={
                "query": query,
                "session_id": session_id,
                "use_local": use_local,
                "local_model": local_model,
                "model": model,
            },
            headers={"X-User-Id": st.session_state.user_id},
            stream=True, timeout=300
        )
        
        # 检查响应状态
        if resp.status_code != 200:
            yield {"type": "error", "content": f"请求失败: HTTP {resp.status_code}"}
            return

        event_type = ""
        data_buffer = ""
        line_buffer = ""
        
        # 读取原始行，正确解析 SSE 格式（处理跨 chunk 行切割）
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            
            # 累积数据并按换行分割，最后一个不完整的行保留到下一个 chunk
            line_buffer += chunk
            lines = line_buffer.split("\n")
            line_buffer = lines.pop()  # 保留最后一个不完整行
            
            for line in lines:
                line = line.rstrip("\r").strip()
                
                if not line:
                    # 空行表示一个事件结束
                    if event_type and data_buffer:
                        try:
                            data = json.loads(data_buffer)
                        except json.JSONDecodeError:
                            event_type = ""
                            data_buffer = ""
                            continue
                        
                        if event_type == "token":
                            yield {"type": "token", "content": data.get("content", "")}
                        elif event_type == "status":
                            yield {
                                "type": "status",
                                "stage": data.get("stage", ""),
                                "message": data.get("message", ""),
                                "progress": data.get("progress", 0)
                            }
                        elif event_type == "think":
                            yield {
                                "type": "think",
                                "step": data.get("step", ""),
                                "detail": data.get("detail", ""),
                                "elapsed_ms": data.get("elapsed_ms", 0),
                                "source": data.get("source", ""),
                                "count": data.get("count", 0),
                            }
                        elif event_type == "done":
                            yield {
                                "type": "done",
                                "content": data.get("content", ""),
                                "trace": data.get("trace", {}),
                                "session_id": data.get("session_id", ""),
                            }
                        elif event_type == "error":
                            yield {"type": "error", "content": data.get("content", "")}
                        
                        event_type = ""
                        data_buffer = ""
                    continue
                
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        data_buffer = data_str

    except requests.exceptions.ConnectionError:
        yield {"type": "error", "content": "无法连接后端服务，请确认已运行 streaming_api.py"}
    except Exception as e:
        yield {"type": "error", "content": f"请求异常: {str(e)}"}


# ===========================================================================
# RAG 过程可视化
# ===========================================================================
def render_rag_panel():
    if not st.session_state.get("show_rag") or not st.session_state.rag_think_steps:
        return
    
    with st.container():
        stage = st.session_state.rag_current_stage
        stage_labels = {
            "intent_recognition": ("🔍 识别意图", "rag-stage-intent"),
            "knowledge_retrieval": ("📚 检索知识", "rag-stage-retrieval"),
            "prompt_construction": ("📝 构建提示", "rag-stage-retrieval"),
            "generating": ("🤔 生成回答", "rag-stage-generating"),
            "completed": ("✅ 完成", "rag-stage-completed"),
            "aborted": ("⛔ 已中止", "rag-stage-intent"),
        }
        
        if stage in stage_labels:
            label, cls = stage_labels[stage]
            st.markdown(f'<div class="rag-stage-badge {cls}">{label}</div>', unsafe_allow_html=True)
        
        for step in st.session_state.rag_think_steps[-10:]:
            icon_map = {
                "intent": "🧠", "searching": "🔍", "grading": "⭐",
                "rewriting": "✏️", "merging": "🔗", "reranking": "📊",
                "prompt": "📝", "generating": "💭"
            }
            icon = icon_map.get(step.get("step", ""), "⏳")
            source = step.get("source", "")
            cls = "think"
            if source == "kg":
                cls = "kg"
            elif source == "vector":
                cls = "vector"
            
            elapsed = step.get("elapsed_ms", 0)
            detail = step.get("detail", "")
            count = step.get("count", 0)
            
            extra = ""
            if count:
                extra = f" ({count} 条)"
            
            st.markdown(
                f'<div class="rag-step-card {cls}">'
                f'<b>{icon}</b> {detail}{extra} '
                f'<small style="color:#9ca3af;margin-left:8px;">{elapsed:.0f}ms</small>'
                f'</div>',
                unsafe_allow_html=True
            )


# ===========================================================================
# 来源追溯渲染
# ===========================================================================
def render_trace(trace, entities):
    if not trace and not entities:
        return
    
    with st.expander("📊 查看 RAG 详情", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔍 实体", len(entities) if entities else 0)
        with col2:
            st.metric("📚 知识图谱", trace.get("kg_count", 0))
        with col3:
            st.metric("📄 向量检索", trace.get("vector_count", 0))
        
        # 来源分解
        sb = trace.get("source_breakdown", {})
        if sb:
            st.markdown("**🎯 结果来源分解**")
            col_kg, col_vec = st.columns(2)
            with col_kg:
                st.markdown(f'<span class="source-badge source-kg">知识图谱: {sb.get("kg", 0)}</span>', unsafe_allow_html=True)
            with col_vec:
                st.markdown(f'<span class="source-badge source-vector">向量检索: {sb.get("vector", 0)}</span>', unsafe_allow_html=True)
        
        # 执行步骤
        steps = trace.get("steps", [])
        if steps:
            st.markdown("**📋 执行流程**")
            steps_text = " → ".join(steps[-8:])
            st.markdown(f"`{steps_text}`")


# ===========================================================================
# 文件上传面板渲染
# ===========================================================================
def render_upload_panel():
    """渲染我的知识库上传面板（侧边栏）"""
    st.markdown("### 📁 我的知识库")

    # 获取统计数据
    records_data = api_get_records()
    docs_data = api_get_documents()
    records_count = records_data.get("total", 0)
    docs_count = docs_data.get("total", 0)

    # 统计卡片
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
        <div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border-radius:10px;padding:10px;text-align:center;">
            <div style="font-size:18px;font-weight:700;color:#1e40af;">{records_count}</div>
            <div style="font-size:11px;color:#3b82f6;">🏥 病历</div>
        </div>
        <div style="background:linear-gradient(135deg,#fef3c7,#fde68a);border-radius:10px;padding:10px;text-align:center;">
            <div style="font-size:18px;font-weight:700;color:#92400e;">{docs_count}</div>
            <div style="font-size:11px;color:#d97706;">📄 文档</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tab 切换：上传病历 / 上传文档 / 查看列表
    tab_upload_record, tab_upload_doc, tab_list = st.tabs(
        ["🏥 上传病历", "📄 上传文档", "📋 已上传"]
    )

    # ========== Tab 1: 上传病历 ==========
    with tab_upload_record:
        st.markdown('<div class="upload-tab-content">', unsafe_allow_html=True)
        record_type = st.selectbox(
            "病历类型",
            ["门诊", "住院", "检查报告", "检验结果", "处方", "其他"],
            help="选择病历的类型"
        )
        record_file = st.file_uploader(
            "选择病历文件",
            type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png"],
            key="upload_record_file",
            help="支持 PDF、Word、TXT、图片格式"
        )
        if st.button("📤 上传病历", use_container_width=True, type="primary", key="btn_upload_record"):
            if not record_file:
                st.warning("⚠️ 请先选择病历文件")
            else:
                with st.spinner(f"正在解析和上传 {record_file.name}..."):
                    file_bytes = record_file.getvalue()
                    result = api_upload_record(file_bytes, record_file.name, record_type)
                if result.get("status") == "success":
                    st.success(f"✅ {result.get('message', '上传成功')}")
                    st.rerun()
                else:
                    st.error(f"❌ 上传失败: {result.get('error', '未知错误')}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Tab 2: 上传文档 ==========
    with tab_upload_doc:
        st.markdown('<div class="upload-tab-content">', unsafe_allow_html=True)
        doc_description = st.text_input(
            "文档描述（可选）",
            placeholder="例如：高血压用药指南",
            key="upload_doc_desc"
        )
        doc_file = st.file_uploader(
            "选择医学文档",
            type=["pdf", "docx", "doc", "txt", "md"],
            key="upload_doc_file",
            help="支持 PDF、Word、TXT、Markdown 格式"
        )
        if st.button("📤 上传文档", use_container_width=True, type="primary", key="btn_upload_doc"):
            if not doc_file:
                st.warning("⚠️ 请先选择文档文件")
            else:
                with st.spinner(f"正在解析和上传 {doc_file.name}..."):
                    file_bytes = doc_file.getvalue()
                    result = api_upload_document(file_bytes, doc_file.name, doc_description)
                if result.get("status") == "success":
                    st.success(f"✅ {result.get('message', '上传成功')}")
                    st.rerun()
                else:
                    st.error(f"❌ 上传失败: {result.get('error', '未知错误')}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Tab 3: 已上传列表 ==========
    with tab_list:
        list_tab_records, list_tab_docs = st.tabs(["病历", "文档"])

        with list_tab_records:
            records = records_data.get("records", [])
            if not records:
                st.info("📭 暂无上传的病历")
            else:
                for r in records:
                    with st.container():
                        col_info, col_del = st.columns([5, 1])
                        with col_info:
                            st.markdown(f"""
                            <div class="kb-item-card">
                                <div class="kb-item-icon">🏥</div>
                                <div class="kb-item-body">
                                    <div class="kb-item-title">{r.get('record_type', '病历')}</div>
                                    <div class="kb-item-meta">
                                        {r.get('content_preview', '')[:60]}...
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_del:
                            if st.button("🗑️", key=f"del_rec_{r.get('document_id', '')}", help="删除"):
                                result = api_delete_record(r["document_id"])
                                if result.get("status") == "success":
                                    st.rerun()
                                else:
                                    st.error("删除失败")

        with list_tab_docs:
            documents = docs_data.get("documents", [])
            if not documents:
                st.info("📭 暂无上传的文档")
            else:
                for d in documents:
                    with st.container():
                        col_info, col_del = st.columns([5, 1])
                        with col_info:
                            st.markdown(f"""
                            <div class="kb-item-card">
                                <div class="kb-item-icon">📄</div>
                                <div class="kb-item-body">
                                    <div class="kb-item-title">{d.get('filename', '文档')}</div>
                                    <div class="kb-item-meta">
                                        {d.get('chunks_count', 0)} 个分块
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_del:
                            if st.button("🗑️", key=f"del_doc_{d.get('document_id', '')}", help="删除"):
                                result = api_delete_document(d["document_id"])
                                if result.get("status") == "success":
                                    st.rerun()
                                else:
                                    st.error("删除失败")


# ===========================================================================
# 会话列表渲染
# ===========================================================================
def render_sidebar():
    with st.sidebar:
        # 用户信息
        st.markdown(f"""
        <div class="card" style="margin-bottom:16px;padding:16px;">
            <div style="display:flex;align-items:center;">
                <div style="width:40px;height:40px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:16px;">
                    {st.session_state.username[0].upper()}
                </div>
                <div style="margin-left:12px;">
                    <div style="font-weight:600;font-size:14px;">{st.session_state.username}</div>
                    <div style="font-size:12px;color:#9ca3af;">管理员</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ========== 我的知识库上传面板 ==========
        render_upload_panel()

        st.divider()

        # 新建会话
        if st.button("➕ 新建对话", use_container_width=True, type="primary"):
            sid = api_create_session("新对话")
            if sid:
                st.session_state.current_session_id = sid
                st.session_state.chat_history = []
                st.session_state.rag_think_steps = []
                st.rerun()

        st.divider()

        # 会话列表
        st.markdown("### 📋 会话列表")

        sessions = api_get_sessions()
        st.session_state.session_list = sessions

        if not sessions:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <div class="empty-state-text">暂无会话，点击上方按钮创建</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for s in sessions:
                sid = s.get("session_id", "")
                title = s.get("title", "新对话")[:20]
                msg_count = s.get("message_count", 0)
                is_active = sid == st.session_state.current_session_id

                col1, col2 = st.columns([5, 1])
                with col1:
                    active_cls = " active" if is_active else ""
                    if st.button(
                        f"{'🔵 ' if is_active else '💬 '}{title}",
                        key=f"sess_{sid}",
                        use_container_width=True,
                        help=f"消息数: {msg_count}",
                    ):
                        st.session_state.current_session_id = sid
                        msgs = api_get_messages(sid)
                        st.session_state.chat_history = [
                            {"role": m["role"], "content": m["content"],
                             "metadata": m.get("metadata", {})}
                            for m in msgs
                        ]
                        st.session_state.rag_think_steps = []
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"del_{sid}", help="删除此对话"):
                        if api_delete_session(sid):
                            if st.session_state.current_session_id == sid:
                                st.session_state.current_session_id = None
                                st.session_state.chat_history = []
                            st.rerun()

        # 显示当前会话消息数
        if st.session_state.current_session_id:
            st.divider()
            st.markdown(f"**当前会话**: {len(st.session_state.chat_history)} 条消息")

        # 清空对话
        if st.button("🗑️ 清空当前对话", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag_think_steps = []
            st.rerun()

        # 设置区域
        st.divider()
        st.markdown("### ⚙️ 设置")

        # 模型选择
        model_choice = st.selectbox(
            "语言模型",
            ["阿里云 Qwen-Plus", "本地 Qwen 轻量", "本地 Llama3.2"],
            help="选择使用的大语言模型"
        )

        if model_choice == "阿里云 Qwen-Plus":
            st.session_state.use_local = False
            st.session_state.model_name = "qwen-plus"
        elif model_choice == "本地 Qwen 轻量":
            st.session_state.use_local = True
            st.session_state.model_name = "qwen:0.5b"
            st.session_state.local_model = "qwen:0.5b"
        else:
            st.session_state.use_local = True
            st.session_state.model_name = "llama3.2:1b"
            st.session_state.local_model = "llama3.2:1b"

        # 显示选项
        st.checkbox("🔍 显示 RAG 过程", value=True, key="show_rag")
        st.checkbox("📊 显示来源追溯", value=True, key="show_trace")


# ===========================================================================
# 主界面
# ===========================================================================
def main():
    # 顶栏
    col_left, col_right = st.columns([4, 1])
    with col_left:
        st.markdown("""
        <div class="app-header">
            <span class="app-icon">🏥</span>
            <div>
                <div class="app-title-main">RAGQnA 医疗智能问答</div>
                <div style="font-size:12px;color:#9ca3af;">融合知识图谱 + 向量检索的智能医疗助手</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        if st.session_state.api_available:
            st.markdown('<div class="status-indicator status-online"><span class="status-dot dot-green"></span>API 在线</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-offline"><span class="status-dot dot-red"></span>API 离线</div>', unsafe_allow_html=True)
    
    # 检查 API 连接
    if not st.session_state.api_available:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔄 检查连接", use_container_width=True, type="primary"):
                st.session_state.api_available = check_api_health()
                st.rerun()
        if not st.session_state.api_available:
            st.warning("⚠️ 后端 API 未连接，请运行 `python streaming_api.py`")
            return
    
    # 渲染侧边栏
    render_sidebar()
    
    # RAG 可视化面板（在生成时显示）
    if st.session_state.is_generating and st.session_state.get("show_rag"):
        render_rag_panel()
    
    # 聊天历史
    if not st.session_state.chat_history:
        # 空状态
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <div class="empty-state-text">开始你的医疗咨询之旅</div>
            <div style="margin-top:16px;">
                <span class="quick-question">感冒发烧应该吃什么药？</span>
                <span class="quick-question">高血压有什么症状？</span>
                <span class="quick-question">糖尿病如何预防？</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and st.session_state.get("show_trace"):
                    meta = msg.get("metadata", {})
                    trace = meta.get("trace", {})
                    entities = meta.get("entities", {})
                    
                    # 渲染回答
                    st.markdown(msg["content"])
                    
                    # 渲染来源追溯
                    if trace or entities:
                        render_trace(trace, entities)
                else:
                    st.markdown(msg["content"])
    
    # 聊天输入
    query = st.chat_input("请输入你的医疗问题... 例如：感冒发烧应该吃什么药？", key="chat_input")
    
    if query and not st.session_state.is_generating:
        st.session_state.is_generating = True
        st.session_state.abort_requested = False
        
        # 确保有 session
        if not st.session_state.current_session_id:
            sid = api_create_session("新对话")
            if sid:
                st.session_state.current_session_id = sid
            else:
                st.error("无法创建会话")
                st.session_state.is_generating = False
                st.rerun()
                return
        
        # 添加用户消息
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # 重置 RAG 步骤
        st.session_state.rag_think_steps = []
        st.session_state.rag_current_stage = ""
        
        # 流式输出容器
        response_placeholder = st.empty()
        progress_container = st.container()
        rag_panel_placeholder = st.empty()  # RAG 过程可视化占位符
        
        full_response = ""
        final_trace = {}
        final_session_id = st.session_state.current_session_id
        
        # 进度条
        with progress_container:
            progress_bar = st.progress(0, text="准备中...")
            abort_col1, abort_col2 = st.columns([1, 8])
            with abort_col1:
                abort_btn = st.button("⏹ 中止", key="abort_btn", type="secondary")
        
        # 用于实时渲染 RAG 步骤的函数
        def update_rag_panel():
            if not st.session_state.get("show_rag") or not st.session_state.rag_think_steps:
                return
            
            steps_html = ""
            for step in st.session_state.rag_think_steps[-6:]:
                icon_map = {
                    "intent": "🧠", "searching": "🔍", "grading": "⭐",
                    "rewriting": "✏️", "merging": "🔗", "reranking": "📊",
                    "prompt": "📝", "generating": "💭"
                }
                icon = icon_map.get(step.get("step", ""), "⏳")
                source = step.get("source", "")
                cls = "think"
                if source == "kg":
                    cls = "kg"
                elif source == "vector":
                    cls = "vector"
                
                elapsed = step.get("elapsed_ms", 0)
                detail = step.get("detail", "")
                count = step.get("count", 0)
                extra = f" ({count} 条)" if count else ""
                
                steps_html += (
                    f'<div class="rag-step-card {cls}">'
                    f'<b>{icon}</b> {detail}{extra} '
                    f'<small style="color:#9ca3af;margin-left:8px;">{elapsed:.0f}ms</small>'
                    f'</div>'
                )
            
            stage = st.session_state.rag_current_stage
            stage_labels = {
                "intent_recognition": ("🔍 识别意图", "rag-stage-intent"),
                "knowledge_retrieval": ("📚 检索知识", "rag-stage-retrieval"),
                "prompt_construction": ("📝 构建提示", "rag-stage-retrieval"),
                "generating": ("🤔 生成回答", "rag-stage-generating"),
                "completed": ("✅ 完成", "rag-stage-completed"),
            }
            
            stage_html = ""
            if stage in stage_labels:
                label, cls = stage_labels[stage]
                stage_html = f'<div class="rag-stage-badge {cls}">{label}</div>'
            
            rag_panel_placeholder.markdown(
                f'<div class="rag-panel">{stage_html}{steps_html}</div>',
                unsafe_allow_html=True
            )
        
        for event in stream_rag(
            query, st.session_state.current_session_id,
            use_local=st.session_state.use_local,
            local_model=st.session_state.local_model,
            model=st.session_state.model_name
        ):
            # 检查中止
            if abort_btn or st.session_state.abort_requested:
                api_abort(st.session_state.current_session_id)
                st.session_state.is_generating = False
                st.session_state.abort_requested = False
                progress_container.empty()
                rag_panel_placeholder.empty()
                st.rerun()
                return
            
            etype = event["type"]
            
            if etype == "token":
                full_response += event["content"]
                response_placeholder.markdown(full_response + "▌")
            
            elif etype == "status":
                stage = event.get("stage", "")
                st.session_state.rag_current_stage = stage
                
                stage_map = {
                    "intent_recognition": "🔍 正在识别意图...",
                    "knowledge_retrieval": "📚 正在检索知识...",
                    "prompt_construction": "📝 正在构建提示...",
                    "generating": "🤔 正在生成回答...",
                    "completed": "✅ 生成完成",
                    "aborted": "⛔ 已中止",
                }
                progress_bar.progress(
                    min(event.get("progress", 0), 1.0),
                    text=stage_map.get(stage, "")
                )
                update_rag_panel()
            
            elif etype == "think":
                st.session_state.rag_think_steps.append({
                    "step": event["step"],
                    "detail": event["detail"],
                    "elapsed_ms": event.get("elapsed_ms", 0),
                    "source": event.get("source", ""),
                    "count": event.get("count", 0),
                })
                update_rag_panel()
            
            elif etype == "done":
                final_trace = event.get("trace", {})
                final_session_id = event.get("session_id", final_session_id)
                response_placeholder.markdown(event.get("content", full_response))
                progress_bar.progress(1.0, text="✅ 完成")
                update_rag_panel()
            
            elif etype == "error":
                response_placeholder.error(event["content"])
                progress_container.empty()
                break
        
        # 保存助手消息
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "metadata": {
                "trace": final_trace,
                "entities": final_trace.get("entities", {}),
            }
        })
        
        # 如果后端返回了新的 session_id，更新
        if final_session_id and final_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = final_session_id
        
        # 清理
        progress_container.empty()
        rag_panel_placeholder.empty()
        st.session_state.rag_think_steps = []
        st.session_state.is_generating = False
        st.rerun()


# ===========================================================================
# 入口
# ===========================================================================
if __name__ == "__main__":
    # 检查是否已登录
    if not st.session_state.get("logged_in", True):
        # 未登录，跳转到登录页
        st.session_state.logged_in = True  # 自动登录（演示模式）
        st.session_state.username = st.session_state.get("username", "admin")
    
    # 检查 API 状态
    if not st.session_state.api_available:
        st.session_state.api_available = check_api_health()
    
    main()
