"""IPL Cricket Prediction Bot — FastAPI Application."""

import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from src.database import (
    get_venue_stats, get_head_to_head, get_all_teams,
    get_all_venues, get_connection, get_match_features,
    close_connection, compute_team_strengths,
)
from src.models import (
    MatchPredictionRequest, TossImpactRequest, PlayerPerformanceRequest,
    WinnerPrediction, TossImpactResponse, VenueStatsResponse,
    HeadToHeadResponse, PlayerPerformanceResponse, HealthResponse,
)
from src.predictor import predictor
from src.auth import APIKeyMiddleware
from src.cache import cache_get, cache_set
from src.metrics import MetricsMiddleware, metrics_endpoint, PREDICTION_COUNT, MODELS_LOADED, DB_MATCHES, CACHE_HITS, CACHE_MISSES
from src.websocket import ws_predictions, manager

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting IPL Prediction Bot v%s", VERSION)

    # Load models
    predictor.load_models()
    MODELS_LOADED.set(len(predictor.models))

    # Compute team strengths from actual data
    try:
        dynamic_strengths = compute_team_strengths()
        if dynamic_strengths:
            predictor.update_team_strengths(dynamic_strengths)
            logger.info("Team strengths computed from %d teams", len(dynamic_strengths))
    except Exception as e:
        logger.warning("Could not compute dynamic strengths: %s", e)

    # Set DB metrics
    try:
        conn = get_connection()
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        DB_MATCHES.set(total)
    except Exception:
        pass

    yield

    logger.info("Shutting down IPL Prediction Bot")
    close_connection()


# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🏏 IPL Cricket Prediction Bot",
    description="""
Predicts IPL match outcomes using XGBoost models trained on historical T20 data.

## Features
- **Match Winner Prediction** — XGBoost classifier with 57.9% accuracy
- **Toss Impact Analysis** — quantifies toss decision advantage per venue
- **Player Performance** — composite performance scoring by role
- **Venue & Head-to-Head Stats** — historical data from 1,200+ matches
- **WebSocket** — real-time predictions via `ws://host/ws/predictions`
- **Prometheus Metrics** — available at `/metrics`

## Authentication
Set `API_KEY` env var to enable API key auth. Pass via `X-API-Key` header.
Leave unset for open access (public endpoints: `/`, `/health`, `/docs`, `/metrics`).
    """,
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware order matters — outermost first
app.add_middleware(MetricsMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Custom OpenAPI Schema ────────────────────────────────────────────────────

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["info"]["x-logo"] = {"url": "https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/200px-Indian_Premier_League_Official_Logo.svg.png"}
    schema["info"]["contact"] = {"name": "IPL Bot API", "url": "https://github.com/LOKii1612/Ipl_prediction"}
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi


# ── Error handler ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": request.url.path},
    )


# ── Prediction Endpoints ─────────────────────────────────────────────────────

@app.post("/predict/match", response_model=WinnerPrediction, tags=["Predictions"],
          summary="Predict match winner",
          responses={200: {"description": "Match prediction with probabilities and key factors"}})
async def predict_match(req: MatchPredictionRequest):
    """Predict the winner of an IPL match given teams, venue, toss winner, and toss decision."""
    if req.toss_winner not in (req.team1, req.team2):
        raise HTTPException(status_code=422, detail="toss_winner must be one of team1 or team2")

    # Check cache
    cache_params = req.model_dump()
    cached = cache_get("match", cache_params)
    if cached:
        CACHE_HITS.inc()
        return cached
    CACHE_MISSES.inc()

    try:
        db_feat = get_match_features(req.team1, req.team2, req.venue, req.toss_winner, req.toss_decision)
    except Exception:
        db_feat = None

    result = predictor.predict_match_winner(req.team1, req.team2, req.venue, req.toss_winner, req.toss_decision, db_feat)
    PREDICTION_COUNT.labels(model="match_winner").inc()

    cache_set("match", cache_params, result)
    return result


@app.post("/predict/toss-impact", response_model=TossImpactResponse, tags=["Predictions"],
          summary="Analyse toss impact",
          responses={200: {"description": "Toss decision analysis with win probabilities"}})
async def predict_toss_impact(req: TossImpactRequest):
    """Analyse how a toss decision affects match outcome at a given venue."""
    cache_params = req.model_dump()
    cached = cache_get("toss", cache_params)
    if cached:
        CACHE_HITS.inc()
        return cached
    CACHE_MISSES.inc()

    result = predictor.predict_toss_impact(req.venue, req.toss_decision, req.team1, req.team2)
    PREDICTION_COUNT.labels(model="toss_impact").inc()

    cache_set("toss", cache_params, result)
    return result


@app.post("/predict/player-performance", response_model=PlayerPerformanceResponse, tags=["Predictions"],
          summary="Predict player performance",
          responses={200: {"description": "Player performance score and tier"}})
async def predict_player_performance(req: PlayerPerformanceRequest):
    """Predict a player's performance score for a given match context."""
    cache_params = req.model_dump()
    cached = cache_get("player", cache_params)
    if cached:
        CACHE_HITS.inc()
        return cached
    CACHE_MISSES.inc()

    result = predictor.predict_player_performance(req.player_name, req.team, req.venue, req.role)
    PREDICTION_COUNT.labels(model="player_performance").inc()

    cache_set("player", cache_params, result)
    return result


# ── Stats Endpoints ──────────────────────────────────────────────────────────

@app.get("/stats/venue/{venue}", response_model=VenueStatsResponse, tags=["Stats"],
         summary="Venue statistics",
         responses={404: {"description": "Venue not found"}})
async def venue_stats(venue: str):
    """Return historical statistics for a given venue (partial name match, case-insensitive)."""
    stats = get_venue_stats(venue)
    if not stats:
        raise HTTPException(status_code=404, detail=f"No data found for venue: {venue}")

    bf = stats["batting_first_win_pct"]
    rec = "Bat first" if bf >= 50 else "Chase"
    stats["recommendation"] = f"{rec} (batting-first wins {bf:.1f}% here)"
    return stats


@app.get("/stats/head-to-head/{team1}/{team2}", response_model=HeadToHeadResponse, tags=["Stats"],
         summary="Head-to-head records",
         responses={404: {"description": "No matches found between teams"}})
async def head_to_head(team1: str, team2: str):
    """Return head-to-head records between two IPL teams."""
    data = get_head_to_head(team1.upper(), team2.upper())
    if data["total_matches"] == 0:
        raise HTTPException(status_code=404, detail=f"No head-to-head data for {team1} vs {team2}")

    dominant = team1 if data["team1_wins"] > data["team2_wins"] else team2
    if data["team1_wins"] == data["team2_wins"]:
        dominant = "Equal"
    data["dominant_team"] = dominant
    return data


@app.get("/stats/teams", tags=["Stats"], summary="List all teams")
@app.get("/teams", tags=["Stats"], include_in_schema=False)
async def list_teams():
    """List all IPL teams in the database."""
    return {"teams": get_all_teams()}


@app.get("/stats/venues", tags=["Stats"], summary="List all venues")
@app.get("/venues", tags=["Stats"], include_in_schema=False)
async def list_venues():
    """List all venues in the database."""
    return {"venues": get_all_venues()}


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/predictions")
async def websocket_predictions(ws: WebSocket):
    """
    WebSocket for real-time predictions.

    Send JSON:
    ```json
    {"type": "predict_match", "team1": "MI", "team2": "CSK",
     "venue": "Wankhede Stadium, Mumbai", "toss_winner": "MI", "toss_decision": "bat"}
    ```
    """
    await ws_predictions(ws, predictor)


# ── Metrics ──────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["System"], summary="Prometheus metrics", include_in_schema=False)
async def metrics():
    """Prometheus-format metrics endpoint."""
    return await metrics_endpoint()


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
async def health_check():
    """Health check — confirms models loaded and DB accessible."""
    db_ok = False
    total = 0
    teams: list[str] = []
    try:
        conn = get_connection()
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        rows = conn.execute(
            "SELECT DISTINCT team1 FROM matches UNION SELECT DISTINCT team2 FROM matches ORDER BY 1"
        ).fetchall()
        teams = [r[0] for r in rows]
        db_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if (predictor.is_loaded and db_ok) else "degraded",
        "models_loaded": predictor.is_loaded,
        "database_connected": db_ok,
        "total_matches_in_db": total,
        "available_teams": teams,
        "version": VERSION,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "🏏 IPL Cricket Prediction Bot",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "websocket": "/ws/predictions",
    }
