from __future__ import annotations

import hmac
import os

from fastapi import WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


TOKEN_HEADER = "X-Vision-Session"
WS_PROTOCOL = "vision-session"


def desktop_token() -> str | None:
    if os.getenv("VIS_DESKTOP_MODE") != "1":
        return None
    return os.getenv("VIS_API_TOKEN") or None


def valid_token(candidate: str | None) -> bool:
    expected = desktop_token()
    return expected is None or bool(candidate) and hmac.compare_digest(candidate, expected)


class DesktopSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "OPTIONS" and not valid_token(request.headers.get(TOKEN_HEADER)):
            return JSONResponse({"detail": "Invalid desktop session"}, status_code=401)
        return await call_next(request)


def websocket_protocol(websocket: WebSocket) -> str | None:
    if desktop_token() is None:
        return None
    if valid_token(websocket.headers.get(TOKEN_HEADER)):
        return None
    protocols = [value.strip() for value in websocket.headers.get("sec-websocket-protocol", "").split(",")]
    if len(protocols) >= 2 and protocols[0] == WS_PROTOCOL and valid_token(protocols[1]):
        return WS_PROTOCOL
    return ""
