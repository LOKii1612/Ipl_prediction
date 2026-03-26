"""Load trained XGBoost models and produce predictions (thread-safe)."""

import json
import os
import threading
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent / "models"))

# Venue batting-first win rate lookup (fallback when DB has no history)
VENUE_BF_RATES: dict[str, float] = {
    "wankhede": 0.47,
    "chidambaram": 0.44,
    "eden": 0.50,
    "chinnaswamy": 0.52,
    "rajiv gandhi": 0.48,
    "arun jaitley": 0.49,
    "punjab cricket": 0.53,
    "sawai mansingh": 0.46,
    "narendra modi": 0.55,
    "ekana": 0.48,
    "maharashtra": 0.51,
    "brabourne": 0.50,
    "dy patil": 0.52,
    "sharjah": 0.43,
    "dubai": 0.44,
    "abu dhabi": 0.46,
    "holkar": 0.54,
}

TEAM_STRENGTHS: dict[str, float] = {
    "MI": 0.62, "CSK": 0.61, "GT": 0.58, "KKR": 0.53,
    "SRH": 0.52, "RR": 0.51, "DC": 0.48, "RCB": 0.47,
    "LSG": 0.49, "PBKS": 0.46, "KXIP": 0.44, "DD": 0.43,
    "RPS": 0.46, "PWI": 0.35,
}

# Default batting/bowling stats per role (used when model needs player features)
_ROLE_BATTING_AVG   = {"batsman": 35.0, "bowler": 5.0,   "allrounder": 22.0}
_ROLE_STRIKE_RATE   = {"batsman": 130.0, "bowler": 0.0,  "allrounder": 125.0}
_ROLE_BOWL_ECONOMY  = {"batsman": 0.0,  "bowler": 7.5,   "allrounder": 8.5}

# Season norm for current/recent matches (2024 = (2024-2008)/16 ≈ 1.0)
_CURRENT_SEASON_NORM = 1.0


class IPLPredictor:
    def __init__(self):
        self.models: dict[str, xgb.XGBModel] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def load_models(self) -> bool:
        model_files = {
            "match_winner": "match_winner_model.json",
            "toss_impact": "toss_impact_model.json",
            "batting_first": "batting_first_model.json",
            "player_performance": "player_performance_model.json",
            "venue_advantage": "venue_advantage_model.json",
        }
        loaded = 0
        for name, fname in model_files.items():
            path = MODELS_DIR / fname
            if path.exists():
                try:
                    if name in ("player_performance", "venue_advantage"):
                        m = xgb.XGBRegressor()
                    else:
                        m = xgb.XGBClassifier()
                    m.load_model(str(path))
                    self.models[name] = m
                    loaded += 1
                    logger.info("Loaded model: %s", name)
                except Exception as e:
                    logger.warning("Could not load %s: %s", name, e)
        self._loaded = loaded > 0
        logger.info("Loaded %d/%d models", loaded, len(model_files))
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def update_team_strengths(self, strengths: dict[str, float]) -> None:
        """Update team strengths from data-derived values."""
        TEAM_STRENGTHS.update(strengths)
        logger.info("Updated %d team strengths from data", len(strengths))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _venue_bf_rate(self, venue: str) -> float:
        venue_lower = venue.lower()
        for key, rate in VENUE_BF_RATES.items():
            if key in venue_lower:
                return rate
        return 0.50

    def _team_strength(self, team: str) -> float:
        return TEAM_STRENGTHS.get(team.upper(), 0.50)

    def _confidence_label(self, prob: float) -> str:
        if prob >= 0.72:
            return "High"
        if prob >= 0.60:
            return "Medium"
        return "Low"

    # ── predictions ──────────────────────────────────────────────────────────

    def predict_match_winner(self, team1: str, team2: str, venue: str,
                             toss_winner: str, toss_decision: str,
                             db_features: Optional[dict] = None) -> dict:
        t1_wr = db_features["team1_win_rate"] if db_features else self._team_strength(team1)
        t2_wr = db_features["team2_win_rate"] if db_features else self._team_strength(team2)
        h2h   = db_features.get("h2h_team1_win_rate", 0.5) if db_features else 0.5
        tw_t1 = (db_features.get("toss_winner_is_team1", 1 if toss_winner == team1 else 0)
                 if db_features else (1 if toss_winner == team1 else 0))
        td_bat = 1 if toss_decision.lower() == "bat" else 0
        bf_wr  = (db_features.get("venue_bf_win_rate", self._venue_bf_rate(venue))
                  if db_features else self._venue_bf_rate(venue))

        # strength_diff and season_norm match training features (train_models.py)
        strength_diff = self._team_strength(team1) - self._team_strength(team2)

        # Feature vector: [t1_hist_win_rate, t2_hist_win_rate, h2h_t1_rate,
        #                   toss_winner_is_t1, toss_decision_bat,
        #                   venue_bf_rate, strength_diff, season_norm]
        X = np.array([[t1_wr, t2_wr, h2h, tw_t1, td_bat,
                       bf_wr, strength_diff, _CURRENT_SEASON_NORM]])

        if "match_winner" in self.models:
            with self._lock:
                prob = float(self.models["match_winner"].predict_proba(X)[0][1])
        else:
            # Fallback: logistic-style estimate
            raw = t1_wr - t2_wr + 0.03 * tw_t1 + 0.02 * (td_bat * (bf_wr - 0.5))
            prob = float(1 / (1 + np.exp(-raw * 5 - 0.1)))

        prob = max(0.30, min(0.70, prob))
        winner = team1 if prob >= 0.5 else team2
        win_prob = prob if prob >= 0.5 else 1 - prob

        factors = []
        if t1_wr > t2_wr + 0.05:
            factors.append(f"{team1} has stronger overall record")
        elif t2_wr > t1_wr + 0.05:
            factors.append(f"{team2} has stronger overall record")
        if h2h > 0.6:
            factors.append(f"{team1} dominates head-to-head")
        elif h2h < 0.4:
            factors.append(f"{team2} dominates head-to-head")
        if tw_t1 and td_bat and bf_wr > 0.52:
            factors.append(f"Toss advantage: batting first at this venue wins {bf_wr*100:.0f}%")
        elif tw_t1 and not td_bat and bf_wr < 0.48:
            factors.append("Toss advantage: chasing team historically strong here")
        if not factors:
            factors.append("Close contest expected — teams evenly matched")

        return {
            "predicted_winner": winner,
            "win_probability": round(win_prob, 3),
            "confidence": self._confidence_label(win_prob),
            "team1": team1,
            "team2": team2,
            "team1_win_probability": round(prob, 3),
            "team2_win_probability": round(1 - prob, 3),
            "key_factors": factors,
        }

    def predict_toss_impact(self, venue: str, toss_decision: str,
                            team1: Optional[str] = None,
                            team2: Optional[str] = None) -> dict:
        bf_wr  = self._venue_bf_rate(venue)
        td_bat = 1 if toss_decision.lower() == "bat" else 0
        t1_wr  = self._team_strength(team1) if team1 else 0.50
        t2_wr  = self._team_strength(team2) if team2 else 0.50

        # Feature vector: [toss_decision_bat, venue_bf_rate, season_norm,
        #                   t1_hist_win_rate, t2_hist_win_rate]
        X = np.array([[td_bat, bf_wr, _CURRENT_SEASON_NORM, t1_wr, t2_wr]])

        if "toss_impact" in self.models:
            with self._lock:
                raw_prob = float(self.models["toss_impact"].predict_proba(X)[0][1])
        else:
            raw_prob = bf_wr if td_bat else (1 - bf_wr)

        raw_prob = max(0.30, min(0.70, raw_prob))
        chasing_prob = 1 - raw_prob
        rec = "bat" if bf_wr >= 0.50 else "field"
        impact_score = abs(bf_wr - 0.50) * 2

        if impact_score > 0.10:
            analysis = f"Strong venue bias — batting first wins {bf_wr*100:.0f}% here."
        elif impact_score > 0.04:
            analysis = f"Mild venue trend — batting first wins {bf_wr*100:.0f}% here."
        else:
            analysis = "Neutral venue — toss has minimal statistical impact."

        return {
            "venue": venue,
            "toss_decision": toss_decision,
            "batting_first_win_probability": round(raw_prob, 3),
            "chasing_win_probability": round(chasing_prob, 3),
            "recommended_decision": rec,
            "toss_impact_score": round(impact_score, 3),
            "analysis": analysis,
        }

    def predict_player_performance(self, player_name: str, team: str,
                                   venue: str, role: str) -> dict:
        role_lower = role.lower()
        role_enc   = {"batsman": 0, "bowler": 1, "allrounder": 2}.get(role_lower, 0)
        bf_wr      = self._venue_bf_rate(venue)
        team_str   = self._team_strength(team)

        batting_avg  = _ROLE_BATTING_AVG.get(role_lower, 20.0)
        strike_rate  = _ROLE_STRIKE_RATE.get(role_lower, 100.0)
        bowl_economy = _ROLE_BOWL_ECONOMY.get(role_lower, 0.0)

        # Feature vector: [strength_rating, venue_bf_rate, role_enc,
        #                   batting_avg, strike_rate, bowling_economy]
        X = np.array([[team_str, bf_wr, role_enc, batting_avg, strike_rate, bowl_economy]])

        if "player_performance" in self.models:
            with self._lock:
                score = float(self.models["player_performance"].predict(X)[0])
        else:
            base = {"batsman": 45.0, "bowler": 42.0, "allrounder": 48.0}.get(role_lower, 44.0)
            score = base + team_str * 15 + np.random.normal(0, 3)

        score = max(20.0, min(90.0, score))

        if score >= 70:
            tier, impact = "Elite", "Match-winning"
        elif score >= 55:
            tier, impact = "Strong", "High impact"
        elif score >= 40:
            tier, impact = "Average", "Moderate impact"
        else:
            tier, impact = "Below average", "Low impact"

        return {
            "player_name": player_name,
            "team": team,
            "venue": venue,
            "predicted_performance_score": round(score, 1),
            "performance_tier": tier,
            "impact_rating": impact,
        }


# Module-level singleton
predictor = IPLPredictor()
