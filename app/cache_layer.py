"""
Redis 缓存层

缓存策略：
  - 会话列表: 按 user_id 缓存，TTL 5 分钟
  - 会话消息: 按 session_id 缓存最近 50 条，TTL 5 分钟
  - RAG trace: 按 session_id 缓存，TTL 10 分钟

缓存失效：
  - 新建/删除会话 → 清除 user 会话列表缓存
  - 新增消息 → 清除 session 消息缓存
  - 新 trace → 覆盖旧 trace 缓存

降级策略：
  - Redis 不可用时所有 get 返回 None，set 静默忽略
  - 不影响主流程，上层应检测 None 并回退到 PostgreSQL 直读
"""

import json
import os
from typing import List, Dict, Any, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class CacheLayer:
    """Redis 缓存层 — 不可用时自动降级"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or DEFAULT_REDIS_URL
        self._client = None
        self._available = False
        self._connect()

    def _connect(self):
        if not REDIS_AVAILABLE:
            print("[CacheLayer] redis-py 未安装，缓存不可用")
            return
        try:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            self._client.ping()
            self._available = True
            print(f"[CacheLayer] Redis 连接成功: {self.redis_url}")
        except Exception as e:
            self._available = False
            print(f"[CacheLayer] Redis 连接失败（将使用降级模式）: {e}")

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    # ------------------------------------------------------------------
    # 会话列表缓存
    # ------------------------------------------------------------------

    def _session_list_key(self, user_id: str) -> str:
        return f"rag:user:{user_id}:sessions"

    def cache_session_list(
        self, user_id: str, sessions: List[Dict], ttl: int = 300
    ) -> bool:
        """缓存用户会话列表"""
        if not self.available:
            return False
        try:
            key = self._session_list_key(user_id)
            self._client.setex(key, ttl, json.dumps(sessions, ensure_ascii=False, default=str))
            return True
        except Exception as e:
            print(f"[CacheLayer] 缓存会话列表失败: {e}")
            return False

    def get_cached_sessions(self, user_id: str) -> Optional[List[Dict]]:
        """获取缓存的会话列表"""
        if not self.available:
            return None
        try:
            key = self._session_list_key(user_id)
            data = self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"[CacheLayer] 读取会话列表缓存失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 消息缓存
    # ------------------------------------------------------------------

    def _messages_key(self, session_id: str) -> str:
        return f"rag:session:{session_id}:messages"

    def cache_messages(
        self, session_id: str, messages: List[Dict], ttl: int = 300
    ) -> bool:
        """缓存会话消息"""
        if not self.available:
            return False
        try:
            key = self._messages_key(session_id)
            self._client.setex(key, ttl, json.dumps(messages, ensure_ascii=False, default=str))
            return True
        except Exception as e:
            print(f"[CacheLayer] 缓存消息失败: {e}")
            return False

    def get_cached_messages(self, session_id: str) -> Optional[List[Dict]]:
        """获取缓存的消息"""
        if not self.available:
            return None
        try:
            key = self._messages_key(session_id)
            data = self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"[CacheLayer] 读取消息缓存失败: {e}")
            return None

    # ------------------------------------------------------------------
    # RAG Trace 缓存
    # ------------------------------------------------------------------

    def _trace_key(self, session_id: str) -> str:
        return f"rag:session:{session_id}:trace"

    def cache_rag_trace(
        self, session_id: str, trace: Dict[str, Any], ttl: int = 600
    ) -> bool:
        """缓存 RAG trace（用于前端可观测性展示）"""
        if not self.available:
            return False
        try:
            key = self._trace_key(session_id)
            self._client.setex(key, ttl, json.dumps(trace, ensure_ascii=False, default=str))
            return True
        except Exception as e:
            print(f"[CacheLayer] 缓存 RAG trace 失败: {e}")
            return False

    def get_cached_trace(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的 RAG trace"""
        if not self.available:
            return None
        try:
            key = self._trace_key(session_id)
            data = self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"[CacheLayer] 读取 RAG trace 缓存失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 缓存失效
    # ------------------------------------------------------------------

    def invalidate_session(self, user_id: str, session_id: str) -> bool:
        """删除会话时：清除会话列表缓存 + 该会话的消息/trace 缓存"""
        if not self.available:
            return False
        try:
            keys = [
                self._session_list_key(user_id),
                self._messages_key(session_id),
                self._trace_key(session_id),
            ]
            self._client.delete(*keys)
            return True
        except Exception as e:
            print(f"[CacheLayer] 缓存失效失败: {e}")
            return False

    def invalidate_user_sessions(self, user_id: str) -> bool:
        """清除用户会话列表缓存（新开会话/删除会话时调用）"""
        if not self.available:
            return False
        try:
            self._client.delete(self._session_list_key(user_id))
            return True
        except Exception as e:
            print(f"[CacheLayer] 清除会话列表缓存失败: {e}")
            return False

    def invalidate_messages(self, session_id: str) -> bool:
        """清除单个会话的消息缓存（新消息时调用）"""
        if not self.available:
            return False
        try:
            self._client.delete(self._messages_key(session_id))
            return True
        except Exception as e:
            print(f"[CacheLayer] 清除消息缓存失败: {e}")
            return False
