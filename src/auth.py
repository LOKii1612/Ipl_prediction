"""Optional API key authentication middleware."""

import os
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Set API_KEY env var to enable auth. Leave unset to disable.
API_KEY = os.getenv("API_KEY")

# Paths that never require auth
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid X-API-Key header (when API_KEY is set)."""

    async def dispatch(self, request: Request, call_next):
        if not API_KEY:
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/ws"):
            return await call_next(request)

        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if key != API_KEY:
            logger.warning("Rejected request to %s — invalid API key", path)
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
