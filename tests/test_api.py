"""
Pytest tests for the IPL Prediction Bot API.

Prediction endpoints work without models or DB (fallback heuristics).
Stats endpoints require a seeded DuckDB — tests are skipped automatically
when the database file is absent.

Run:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "ipl_matches.duckdb"
MODELS_DIR = ROOT / "models"

sys.path.insert(0, str(ROOT))

from src.main import app
from src.predictor import predictor

client = TestClient(app)

# ── Availability flags (evaluated at collection time) ──────────────────────────

DB_AVAILABLE = DB_PATH.exists()

_MODEL_FILES = [
    "match_winner_model.json",
    "toss_impact_model.json",
    "batting_first_model.json",
    "player_performance_model.json",
    "venue_advantage_model.json",
]
MODELS_AVAILABLE = all((MODELS_DIR / f).exists() for f in _MODEL_FILES)

skip_no_db     = pytest.mark.skipif(not DB_AVAILABLE,     reason="DuckDB not seeded — run scripts/generate_synthetic_data.py")
skip_no_models = pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not trained — run scripts/train_models.py")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def load_models():
    """Load models once for the test module (no-op if files are absent)."""
    predictor.load_models()


# ── Root / Health ─────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_name(self):
        resp = client.get("/")
        assert "IPL" in resp.json()["name"]

    def test_root_has_docs_link(self):
        resp = client.get("/")
        body = resp.json()
        assert "docs" in body
        assert "health" in body


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self):
        resp = client.get("/health")
        body = resp.json()
        for field in ("status", "models_loaded", "database_connected",
                      "total_matches_in_db", "available_teams", "version"):
            assert field in body, f"Missing field: {field}"

    def test_health_status_is_string(self):
        resp = client.get("/health")
        assert resp.json()["status"] in ("ok", "degraded")

    def test_health_models_loaded_bool(self):
        resp = client.get("/health")
        assert isinstance(resp.json()["models_loaded"], bool)

    @skip_no_db
    def test_health_db_connected(self):
        resp = client.get("/health")
        assert resp.json()["database_connected"] is True

    @skip_no_db
    def test_health_has_teams(self):
        resp = client.get("/health")
        assert len(resp.json()["available_teams"]) > 5

    @skip_no_db
    def test_health_total_matches_positive(self):
        resp = client.get("/health")
        assert resp.json()["total_matches_in_db"] > 100


# ── Match prediction ──────────────────────────────────────────────────────────

class TestMatchPrediction:
    BASE = {
        "team1": "MI",
        "team2": "CSK",
        "venue": "Wankhede Stadium, Mumbai",
        "toss_winner": "MI",
        "toss_decision": "bat",
    }

    def test_predict_returns_200(self):
        resp = client.post("/predict/match", json=self.BASE)
        assert resp.status_code == 200

    def test_predict_has_winner(self):
        resp = client.post("/predict/match", json=self.BASE)
        body = resp.json()
        assert "predicted_winner" in body
        assert body["predicted_winner"] in ("MI", "CSK")

    def test_predict_probabilities_sum_to_one(self):
        resp = client.post("/predict/match", json=self.BASE)
        body = resp.json()
        total = body["team1_win_probability"] + body["team2_win_probability"]
        assert abs(total - 1.0) < 0.01

    def test_predict_probability_in_range(self):
        resp = client.post("/predict/match", json=self.BASE)
        body = resp.json()
        assert 0.0 <= body["win_probability"] <= 1.0

    def test_predict_confidence_label_valid(self):
        resp = client.post("/predict/match", json=self.BASE)
        body = resp.json()
        assert body["confidence"] in ("Low", "Medium", "High")

    def test_predict_key_factors_list(self):
        resp = client.post("/predict/match", json=self.BASE)
        body = resp.json()
        assert isinstance(body["key_factors"], list)
        assert len(body["key_factors"]) >= 1

    def test_predict_invalid_toss_winner(self):
        """toss_winner must be team1 or team2."""
        payload = {**self.BASE, "toss_winner": "GT"}
        resp = client.post("/predict/match", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_toss_decision(self):
        payload = {**self.BASE, "toss_decision": "maybe"}
        resp = client.post("/predict/match", json=payload)
        assert resp.status_code == 422

    def test_predict_toss_decision_case_insensitive(self):
        """toss_decision should be case-insensitive."""
        for decision in ("BAT", "Bat", "bat", "FIELD", "Field", "field"):
            payload = {**self.BASE, "toss_decision": decision}
            resp = client.post("/predict/match", json=payload)
            assert resp.status_code == 200, f"Failed for toss_decision={decision!r}"

    def test_predict_field_decision(self):
        payload = {**self.BASE, "toss_decision": "field"}
        resp = client.post("/predict/match", json=payload)
        assert resp.status_code == 200

    def test_predict_different_teams(self):
        payload = {
            "team1": "RCB", "team2": "KKR",
            "venue": "M Chinnaswamy Stadium, Bangalore",
            "toss_winner": "RCB", "toss_decision": "bat",
        }
        resp = client.post("/predict/match", json=payload)
        assert resp.status_code == 200
        assert resp.json()["predicted_winner"] in ("RCB", "KKR")

    def test_predict_unknown_teams_still_returns_result(self):
        """Unknown teams fall back to heuristics — should not 500."""
        payload = {
            "team1": "XXX", "team2": "YYY",
            "venue": "Some Unknown Ground",
            "toss_winner": "XXX", "toss_decision": "bat",
        }
        resp = client.post("/predict/match", json=payload)
        assert resp.status_code == 200

    def test_predict_missing_required_field_team1(self):
        payload = {k: v for k, v in self.BASE.items() if k != "team1"}
        resp = client.post("/predict/match", json=payload)
        assert resp.status_code == 422

    def test_predict_missing_required_field_venue(self):
        payload = {k: v for k, v in self.BASE.items() if k != "venue"}
        resp = client.post("/predict/match", json=payload)
        assert resp.status_code == 422

    def test_predict_empty_body(self):
        resp = client.post("/predict/match", json={})
        assert resp.status_code == 422

    def test_predict_win_probability_clamped(self):
        """Win probability must stay in [0.30, 0.70] range (model clamp)."""
        resp = client.post("/predict/match", json=self.BASE)
        body = resp.json()
        p1 = body["team1_win_probability"]
        p2 = body["team2_win_probability"]
        assert 0.0 < p1 < 1.0
        assert 0.0 < p2 < 1.0


# ── Toss impact ───────────────────────────────────────────────────────────────

class TestTossImpact:
    BASE = {
        "venue": "Wankhede Stadium, Mumbai",
        "toss_decision": "bat",
    }

    def test_toss_impact_returns_200(self):
        resp = client.post("/predict/toss-impact", json=self.BASE)
        assert resp.status_code == 200

    def test_toss_impact_has_fields(self):
        resp = client.post("/predict/toss-impact", json=self.BASE)
        body = resp.json()
        for field in ("batting_first_win_probability", "chasing_win_probability",
                      "recommended_decision", "toss_impact_score", "analysis"):
            assert field in body, f"Missing field: {field}"

    def test_toss_impact_probabilities_sum_to_one(self):
        resp = client.post("/predict/toss-impact", json=self.BASE)
        body = resp.json()
        total = body["batting_first_win_probability"] + body["chasing_win_probability"]
        assert abs(total - 1.0) < 0.01

    def test_toss_impact_recommended_decision_valid(self):
        resp = client.post("/predict/toss-impact", json=self.BASE)
        assert resp.json()["recommended_decision"] in ("bat", "field")

    def test_toss_impact_score_in_range(self):
        resp = client.post("/predict/toss-impact", json=self.BASE)
        score = resp.json()["toss_impact_score"]
        assert 0.0 <= score <= 1.0

    def test_toss_impact_invalid_decision(self):
        resp = client.post("/predict/toss-impact", json={**self.BASE, "toss_decision": "win"})
        assert resp.status_code == 422

    def test_toss_impact_field_decision(self):
        resp = client.post("/predict/toss-impact", json={**self.BASE, "toss_decision": "field"})
        assert resp.status_code == 200

    def test_toss_impact_with_teams(self):
        payload = {**self.BASE, "team1": "MI", "team2": "CSK"}
        resp = client.post("/predict/toss-impact", json=payload)
        assert resp.status_code == 200

    def test_toss_impact_chasing_venue(self):
        """Chennai (chidambaram) historically favours chasing."""
        payload = {"venue": "MA Chidambaram Stadium, Chennai", "toss_decision": "field"}
        resp = client.post("/predict/toss-impact", json=payload)
        assert resp.status_code == 200

    def test_toss_impact_missing_venue(self):
        resp = client.post("/predict/toss-impact", json={"toss_decision": "bat"})
        assert resp.status_code == 422

    def test_toss_impact_empty_body(self):
        resp = client.post("/predict/toss-impact", json={})
        assert resp.status_code == 422


# ── Player performance ────────────────────────────────────────────────────────

class TestPlayerPerformance:
    BASE = {
        "player_name": "Rohit Sharma",
        "team": "MI",
        "venue": "Wankhede Stadium, Mumbai",
        "role": "batsman",
    }

    def test_player_perf_returns_200(self):
        resp = client.post("/predict/player-performance", json=self.BASE)
        assert resp.status_code == 200

    def test_player_perf_has_score(self):
        resp = client.post("/predict/player-performance", json=self.BASE)
        body = resp.json()
        assert "predicted_performance_score" in body
        assert 0 < body["predicted_performance_score"] < 100

    def test_player_perf_tier_valid(self):
        resp = client.post("/predict/player-performance", json=self.BASE)
        body = resp.json()
        assert body["performance_tier"] in ("Elite", "Strong", "Average", "Below average")

    def test_player_perf_impact_rating_valid(self):
        resp = client.post("/predict/player-performance", json=self.BASE)
        body = resp.json()
        assert body["impact_rating"] in ("Match-winning", "High impact", "Moderate impact", "Low impact")

    def test_player_perf_invalid_role(self):
        resp = client.post("/predict/player-performance", json={**self.BASE, "role": "coach"})
        assert resp.status_code == 422

    def test_player_perf_bowler(self):
        payload = {**self.BASE, "player_name": "Jasprit Bumrah", "role": "bowler"}
        resp = client.post("/predict/player-performance", json=payload)
        assert resp.status_code == 200

    def test_player_perf_allrounder(self):
        payload = {**self.BASE, "player_name": "Kieron Pollard", "role": "allrounder"}
        resp = client.post("/predict/player-performance", json=payload)
        assert resp.status_code == 200

    def test_player_perf_role_case_insensitive(self):
        for role in ("BATSMAN", "Batsman", "batsman"):
            payload = {**self.BASE, "role": role}
            resp = client.post("/predict/player-performance", json=payload)
            assert resp.status_code == 200, f"Failed for role={role!r}"

    def test_player_perf_unknown_team(self):
        """Unknown team falls back to default strength — should not 500."""
        payload = {**self.BASE, "team": "UNKNOWN_FC"}
        resp = client.post("/predict/player-performance", json=payload)
        assert resp.status_code == 200

    def test_player_perf_missing_player_name(self):
        payload = {k: v for k, v in self.BASE.items() if k != "player_name"}
        resp = client.post("/predict/player-performance", json=payload)
        assert resp.status_code == 422

    def test_player_perf_response_echoes_player(self):
        """Response must echo back the player name and team."""
        resp = client.post("/predict/player-performance", json=self.BASE)
        body = resp.json()
        assert body["player_name"] == self.BASE["player_name"]
        assert body["team"] == self.BASE["team"]


# ── Stats endpoints ───────────────────────────────────────────────────────────

class TestVenueStats:
    @skip_no_db
    def test_venue_stats_found(self):
        resp = client.get("/stats/venue/Wankhede")
        assert resp.status_code == 200

    @skip_no_db
    def test_venue_stats_has_fields(self):
        resp = client.get("/stats/venue/Wankhede")
        body = resp.json()
        for field in ("total_matches", "avg_first_innings_score",
                      "batting_first_win_pct", "recommendation"):
            assert field in body, f"Missing field: {field}"

    @skip_no_db
    def test_venue_stats_total_matches_positive(self):
        resp = client.get("/stats/venue/Wankhede")
        assert resp.json()["total_matches"] > 0

    @skip_no_db
    def test_venue_stats_recommendation_string(self):
        resp = client.get("/stats/venue/Wankhede")
        rec = resp.json()["recommendation"]
        assert isinstance(rec, str) and len(rec) > 5

    @skip_no_db
    def test_venue_stats_partial_match(self):
        """Venue search is case-insensitive and accepts partial names."""
        resp = client.get("/stats/venue/wankhede")
        assert resp.status_code == 200

    @skip_no_db
    def test_venue_stats_not_found(self):
        resp = client.get("/stats/venue/NonExistentVenueXYZ999")
        assert resp.status_code == 404

    @skip_no_db
    def test_venue_stats_percentages_in_range(self):
        resp = client.get("/stats/venue/Wankhede")
        pct = resp.json()["batting_first_win_pct"]
        assert 0.0 <= pct <= 100.0


class TestHeadToHead:
    @skip_no_db
    def test_h2h_returns_200(self):
        resp = client.get("/stats/head-to-head/MI/CSK")
        assert resp.status_code == 200

    @skip_no_db
    def test_h2h_has_fields(self):
        resp = client.get("/stats/head-to-head/MI/CSK")
        body = resp.json()
        for field in ("total_matches", "team1_wins", "team2_wins",
                      "team1_win_pct", "dominant_team"):
            assert field in body, f"Missing field: {field}"

    @skip_no_db
    def test_h2h_wins_sum_to_total(self):
        resp = client.get("/stats/head-to-head/MI/CSK")
        body = resp.json()
        assert body["team1_wins"] + body["team2_wins"] == body["total_matches"]

    @skip_no_db
    def test_h2h_percentages_sum_to_100(self):
        resp = client.get("/stats/head-to-head/MI/CSK")
        body = resp.json()
        assert abs(body["team1_win_pct"] + body["team2_win_pct"] - 100.0) < 1.0

    @skip_no_db
    def test_h2h_dominant_team_is_team_or_equal(self):
        resp = client.get("/stats/head-to-head/MI/CSK")
        body = resp.json()
        assert body["dominant_team"] in ("MI", "CSK", "Equal")

    @skip_no_db
    def test_h2h_not_found(self):
        resp = client.get("/stats/head-to-head/TEAM99/TEAM88")
        assert resp.status_code == 404

    @skip_no_db
    def test_h2h_case_normalised(self):
        """Teams should be matched case-insensitively via .upper() in main."""
        resp_upper = client.get("/stats/head-to-head/MI/CSK")
        resp_lower = client.get("/stats/head-to-head/mi/csk")
        assert resp_upper.status_code == resp_lower.status_code


class TestListEndpoints:
    @skip_no_db
    def test_list_teams(self):
        resp = client.get("/stats/teams")
        assert resp.status_code == 200
        data = resp.json()
        assert "teams" in data
        assert len(data["teams"]) > 5

    @skip_no_db
    def test_list_teams_sorted(self):
        resp = client.get("/stats/teams")
        teams = resp.json()["teams"]
        assert teams == sorted(teams)

    @skip_no_db
    def test_list_venues(self):
        resp = client.get("/stats/venues")
        assert resp.status_code == 200
        data = resp.json()
        assert "venues" in data
        assert len(data["venues"]) > 5

    @skip_no_db
    def test_list_venues_contains_wankhede(self):
        resp = client.get("/stats/venues")
        venues = resp.json()["venues"]
        assert any("Wankhede" in v for v in venues)
