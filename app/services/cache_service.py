from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import time
import threading

@dataclass
class CacheItem:
    value: Any
    expires_at: float

class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_items: int = 256):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: Dict[str, CacheItem] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            # 雑に上限を超えたら期限切れ掃除 → まだ多ければ古い順に削る
            self._cleanup_locked(now)
            if len(self._store) >= self.max_items:
                # expires_at が早いものから削除
                for k, _ in sorted(self._store.items(), key=lambda kv: kv[1].expires_at)[: max(1, self.max_items // 10)]:
                    self._store.pop(k, None)
            self._store[key] = CacheItem(value=value, expires_at=now + self.ttl)

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            if item.expires_at < now:
                self._store.pop(key, None)
                return None
            return item.value

    def _cleanup_locked(self, now: float) -> None:
        # 期限切れ削除
        expired = [k for k, v in self._store.items() if v.expires_at < now]
        for k in expired:
            self._store.pop(k, None)
