"""
会话持久化存储 — PostgreSQL

表结构：
  chat_sessions  — 会话元数据（session_id, user_id, title, summary, message_count）
  chat_messages  — 消息记录（message_id, session_id, role, content, metadata, token_count）

功能：
  - 会话 CRUD（创建/列表/删除）
  - 消息追加与查询
  - 长对话自动摘要触发判断
"""

import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import psycopg2
import psycopg2.extras


# PostgreSQL 连接配置 — 从 .env 读取
DEFAULT_DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "rag_medical"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}


class SessionStorage:
    """PostgreSQL 会话与消息持久化"""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        self.db_config = db_config or DEFAULT_DB_CONFIG
        self._conn = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            self._conn = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
            )
            self._conn.autocommit = True
            return self._ensure_tables()
        except Exception as e:
            print(f"[SessionStorage] 连接 PostgreSQL 失败: {e}")
            return False

    def _ensure_tables(self) -> bool:
        """创建表（如果不存在）"""
        ddl = """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id   UUID PRIMARY KEY,
            user_id      VARCHAR(64) NOT NULL,
            title        VARCHAR(256) DEFAULT '新对话',
            summary      TEXT,
            message_count INT DEFAULT 0,
            created_at   TIMESTAMPTZ DEFAULT NOW(),
            updated_at   TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id   UUID PRIMARY KEY,
            session_id   UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            role         VARCHAR(16) NOT NULL,
            content      TEXT NOT NULL,
            metadata     JSONB DEFAULT '{}',
            token_count  INT DEFAULT 0,
            created_at   TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON chat_messages(session_id, created_at);

        CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON chat_sessions(user_id, updated_at DESC);
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(ddl)
            return True
        except Exception as e:
            print(f"[SessionStorage] 建表失败: {e}")
            return False

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # 会话操作
    # ------------------------------------------------------------------

    def create_session(
        self, user_id: str, title: str = "新对话"
    ) -> Optional[str]:
        """创建新会话，返回 session_id"""
        session_id = str(uuid.uuid4())
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO chat_sessions (session_id, user_id, title)
                       VALUES (%s, %s, %s)""",
                    (session_id, user_id, title),
                )
            return session_id
        except Exception as e:
            print(f"[SessionStorage] 创建会话失败: {e}")
            return None

    def get_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有会话列表（按更新时间倒序）"""
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """SELECT session_id, user_id, title, summary, message_count,
                              created_at, updated_at
                       FROM chat_sessions
                       WHERE user_id = %s
                       ORDER BY updated_at DESC""",
                    (user_id,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "session_id": r["session_id"],
                        "user_id": r["user_id"],
                        "title": r["title"],
                        "summary": r["summary"],
                        "message_count": r["message_count"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else "",
                        "updated_at": r["updated_at"].isoformat() if r["updated_at"] else "",
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"[SessionStorage] 获取会话列表失败: {e}")
            return []

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取单个会话信息"""
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM chat_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "session_id": row["session_id"],
                        "user_id": row["user_id"],
                        "title": row["title"],
                        "summary": row["summary"],
                        "message_count": row["message_count"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
                    }
            return None
        except Exception as e:
            print(f"[SessionStorage] 获取会话失败: {e}")
            return None

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """删除会话（仅所有者可删除）"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_sessions WHERE session_id = %s AND user_id = %s",
                    (session_id, user_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            print(f"[SessionStorage] 删除会话失败: {e}")
            return False

    def update_title(self, session_id: str, title: str) -> bool:
        """更新会话标题"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """UPDATE chat_sessions
                       SET title = %s, updated_at = NOW()
                       WHERE session_id = %s""",
                    (title, session_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            print(f"[SessionStorage] 更新标题失败: {e}")
            return False

    def update_summary(self, session_id: str, summary: str) -> bool:
        """更新会话摘要"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """UPDATE chat_sessions
                       SET summary = %s, updated_at = NOW()
                       WHERE session_id = %s""",
                    (summary, session_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            print(f"[SessionStorage] 更新摘要失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 消息操作
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: int = 0,
    ) -> Optional[str]:
        """添加消息，返回 message_id"""
        message_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO chat_messages (message_id, session_id, role, content, metadata, token_count)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (message_id, session_id, role, content, meta_json, token_count),
                )
                # 更新 session 的消息计数和时间
                cur.execute(
                    """UPDATE chat_sessions
                       SET message_count = message_count + 1, updated_at = NOW()
                       WHERE session_id = %s""",
                    (session_id,),
                )
            return message_id
        except Exception as e:
            print(f"[SessionStorage] 添加消息失败: {e}")
            return None

    def get_messages(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取会话消息列表（按时间正序）"""
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """SELECT message_id, session_id, role, content, metadata, token_count, created_at
                       FROM chat_messages
                       WHERE session_id = %s
                       ORDER BY created_at ASC
                       LIMIT %s OFFSET %s""",
                    (session_id, limit, offset),
                )
                rows = cur.fetchall()
                return [
                    {
                        "message_id": r["message_id"],
                        "session_id": r["session_id"],
                        "role": r["role"],
                        "content": r["content"],
                        "metadata": r["metadata"] if isinstance(r["metadata"], dict) else {},
                        "token_count": r["token_count"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else "",
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"[SessionStorage] 获取消息失败: {e}")
            return []

    def get_message_count(self, session_id: str) -> int:
        """获取会话消息数"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE session_id = %s",
                    (session_id,),
                )
                return cur.fetchone()[0]
        except Exception:
            return 0

    def get_recent_messages_for_summary(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取最近的消息用于生成摘要（排除 system 消息）"""
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """SELECT role, content FROM chat_messages
                       WHERE session_id = %s AND role != 'system'
                       ORDER BY created_at DESC
                       LIMIT %s""",
                    (session_id, limit),
                )
                rows = cur.fetchall()
                return [
                    {"role": r["role"], "content": r["content"]}
                    for r in reversed(rows)
                ]
        except Exception as e:
            print(f"[SessionStorage] 获取摘要用消息失败: {e}")
            return []

    def check_summary_needed(
        self, session_id: str, threshold: int = 20
    ) -> bool:
        """判断是否需要触发摘要压缩（消息数超过阈值）"""
        count = self.get_message_count(session_id)
        return count >= threshold

    # ------------------------------------------------------------------
    # 自动摘要
    # ------------------------------------------------------------------

    def try_auto_summarize(
        self,
        session_id: str,
        threshold: int = 20,
        use_local: bool = False,
        local_model: str = "qwen:0.5b",
    ) -> Optional[str]:
        """
        如果消息数超过阈值，生成摘要并更新 session

        Returns:
            生成的摘要文本，或 None（不需要摘要或生成失败）
        """
        if not self.check_summary_needed(session_id, threshold):
            return None

        messages = self.get_recent_messages_for_summary(session_id, 20)
        if len(messages) < 5:
            return None

        summary = self._generate_summary(messages, use_local, local_model)
        if summary:
            self.update_summary(session_id, summary)
        return summary

    def _generate_summary(
        self,
        conversations: List[Dict],
        use_local: bool = False,
        local_model: str = "qwen:0.5b",
    ) -> str:
        """调用 LLM 生成对话摘要"""
        # 构建对话文本
        conv_text = ""
        for msg in conversations[-20:]:
            role_label = "用户" if msg["role"] == "user" else "助手"
            conv_text += f"{role_label}: {msg['content'][:300]}\n"

        prompt = f"""请用不超过100字总结以下医疗对话的核心内容：

{conv_text}

总结："""

        try:
            if use_local:
                import ollama
                result = ollama.generate(model=local_model, prompt=prompt)
                return result["response"].strip()[:200]
            else:
                import dashscope
                from dashscope import Generation
                response = Generation.call(
                    model=Generation.Models.qwen_plus, prompt=prompt
                )
                return response.output.text.strip()[:200]
        except Exception as e:
            print(f"[SessionStorage] 生成摘要失败: {e}")
            return ""
