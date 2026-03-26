"""Optional Redis caching layer for predictions."""

import os
import json
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default

_client = None


def _get_redis():
    """Lazy-init Redis client. Returns None if Redis not configured."""
    global _client
    if _client is not None:
        return _client
    if not REDIS_URL:
        return None
    try:
        import redis
        _client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _client.ping()
        logger.info("Redis connected: %s", REDIS_URL)
        return _client
    except Exception as e:
        logger.warning("Redis unavailable (%s) — caching disabled", e)
        _client = None
        return None


def _cache_key(prefix: str, data: dict) -> str:
    """Deterministic cache key from prefix + sorted dict hash."""
    raw = json.dumps(data, sort_keys=True)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"ipl:{prefix}:{h}"


def cache_get(prefix: str, params: dict) -> Optional[dict]:
    """Try to get cached result. Returns None on miss or if Redis unavailable."""
    r = _get_redis()
    if not r:
        return None
    try:
        key = _cache_key(prefix, params)
        val = r.get(key)
        if val:
            logger.debug("Cache HIT: %s", key)
            return json.loads(val)
        return None
    except Exception:
        return None


def cache_set(prefix: str, params: dict, result: dict) -> None:
    """Store result in cache with TTL."""
    r = _get_redis()
    if not r:
        return
    try:
        key = _cache_key(prefix, params)
        r.setex(key, CACHE_TTL, json.dumps(result))
        logger.debug("Cache SET: %s (ttl=%ds)", key, CACHE_TTL)
    except Exception:
        pass
