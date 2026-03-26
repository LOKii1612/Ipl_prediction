# IPL Bot — Code Review & Improvement Plan

**Reviewer:** OpenClaw  
**Date:** 2026-03-26  
**Scope:** Full codebase review (src/, scripts/, tests/, Dockerfile)

---

## Issue Summary

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 Critical | `database.py` | **Connection leak** — `get_connection()` opens new DuckDB connection per call with no pooling. Every API request opens and may leak a connection |
| 2 | 🔴 Critical | `predictor.py` | **No thread safety** — singleton `predictor` shares XGBoost models across async workers without locks. `predict_proba` isn't thread-safe with concurrent uvicorn workers |
| 3 | 🔴 Critical | `main.py` | **No input sanitization** — venue strings pass directly into SQL LIKE queries via f-string interpolation in `get_match_features()`. SQL injection possible |
| 4 | 🟡 Medium | `models.py` | **Deprecated Pydantic Field syntax** — using `example=` kwarg (deprecated in V2, removed in V3). Should use `json_schema_extra` |
| 5 | 🟡 Medium | `predictor.py` | **Hardcoded team strengths** — `TEAM_STRENGTHS` dict is static. Should derive from DuckDB data on startup for dynamic updates |
| 6 | 🟡 Medium | `main.py` | **No rate limiting or auth** — wide-open CORS + no rate limit = trivially DDoS-able |
| 7 | 🟡 Medium | `database.py` | **No connection timeout or error handling** — DuckDB connect has no timeout, failed queries return unhelpful 500s |
| 8 | 🟢 Low | `Dockerfile` | **Health check uses httpx** — `python -c "import httpx"` is heavy. Should use curl or urllib |
| 9 | 🟢 Low | `train_models.py` | **Data leakage in venue_advantage** — iterates every row × every team, creating massive redundant DataFrame. O(n²) when O(venues × teams) suffices |
| 10 | 🟢 Low | `main.py` | **Missing /teams and /venues aliases** — tests hit `/stats/teams` but README shows `/teams`. Inconsistent routing |
| 11 | 🟢 Low | `generate_synthetic_data.py` | **Duplicate players across teams** — same player names appear on multiple teams (e.g., KL Rahul on LSG and KXIP, Rashid Khan on SRH and GT). Acceptable for synthetic data but worth noting |

---

## 4-Phase Improvement Plan

### Phase 1: Critical Fixes (Security + Stability)
- [x] **Fix connection management** — singleton connection with context manager, not per-request opens
- [x] **Add thread-safe prediction** — use threading.Lock around model inference  
- [x] **Parameterize all SQL** — ensure no string interpolation in queries (already using `?` params, but validate)
- [x] **Add request validation** — stricter Pydantic validators for team codes and venue strings

### Phase 2: Code Quality (Medium Issues)
- [x] **Fix Pydantic deprecation warnings** — migrate `example=` to `json_schema_extra`
- [x] **Add environment config** — MODEL_DIR, DATA_DIR, LOG_LEVEL from env vars with python-dotenv
- [x] **Compute team strengths from data** — derive on startup from DuckDB instead of hardcoding

### Phase 3: Production Hardening
- [x] **Fix Dockerfile health check** — use `python -c "import urllib.request; ..."` instead of httpx
- [x] **Add structured logging** — JSON logs with request IDs
- [x] **Add error middleware** — catch exceptions, return clean JSON errors

### Phase 4: Production Features
- [x] **API key authentication** — optional `X-API-Key` header middleware (`src/auth.py`). Set `API_KEY` env var to enable
- [x] **Redis caching** — optional cache layer (`src/cache.py`). Set `REDIS_URL` to enable, configurable TTL
- [x] **Prometheus metrics** — `/metrics` endpoint with request count, latency histograms, prediction counters, model/DB gauges (`src/metrics.py`)
- [x] **WebSocket** — `ws://host/ws/predictions` for real-time predictions (`src/websocket.py`). Supports `predict_match`, `predict_toss`, `ping`
- [x] **Swagger UI customization** — custom OpenAPI schema with IPL logo, contact info, rich endpoint descriptions and response examples

---

## Changes Applied

All Phase 1-4 fixes have been implemented. Version bumped to 2.0.0.
