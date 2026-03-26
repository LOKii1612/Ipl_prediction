"""WebSocket endpoint for live match prediction updates."""

import json
import logging
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info("WS connected — %d active", len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info("WS disconnected — %d active", len(self.active))

    async def broadcast(self, data: dict):
        """Send JSON payload to all connected clients."""
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.discard(ws)

    @property
    def count(self) -> int:
        return len(self.active)


manager = ConnectionManager()


async def ws_predictions(ws: WebSocket, predictor):
    """
    WebSocket endpoint: client sends match request JSON,
    server replies with prediction result in real-time.

    Message format (send):
    {
        "type": "predict_match",
        "team1": "MI", "team2": "CSK",
        "venue": "Wankhede Stadium, Mumbai",
        "toss_winner": "MI", "toss_decision": "bat"
    }
    """
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                msg_type = data.get("type", "predict_match")

                if msg_type == "predict_match":
                    result = predictor.predict_match_winner(
                        data["team1"], data["team2"], data["venue"],
                        data["toss_winner"], data["toss_decision"],
                    )
                    await ws.send_text(json.dumps({"type": "prediction", **result}))

                elif msg_type == "predict_toss":
                    result = predictor.predict_toss_impact(
                        data["venue"], data["toss_decision"],
                        data.get("team1"), data.get("team2"),
                    )
                    await ws.send_text(json.dumps({"type": "toss_impact", **result}))

                elif msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

                else:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "detail": f"Unknown message type: {msg_type}"
                    }))

            except (KeyError, json.JSONDecodeError) as e:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "detail": str(e),
                }))

    except WebSocketDisconnect:
        manager.disconnect(ws)
