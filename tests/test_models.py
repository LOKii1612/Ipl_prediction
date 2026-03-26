"""
Model validation tests — verify each saved XGBoost model exists and
produces sane predictions.

All tests are automatically skipped if the corresponding model file has
not been trained yet.  Run scripts/train_models.py first.
"""

import pytest
import numpy as np
import xgboost as xgb
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"
sys.path.insert(0, str(ROOT))


# ── Per-model availability helpers ────────────────────────────────────────────

def _model_exists(fname: str) -> bool:
    return (MODELS_DIR / fname).exists()

_ALL_MODEL_FILES = [
    "match_winner_model.json",
    "toss_impact_model.json",
    "batting_first_model.json",
    "player_performance_model.json",
    "venue_advantage_model.json",
]
_ALL_MODELS_AVAILABLE = all(_model_exists(f) for f in _ALL_MODEL_FILES)

skip_if_no_models = pytest.mark.skipif(
    not _ALL_MODELS_AVAILABLE,
    reason="Not all model files present — run scripts/train_models.py"
)


# ── Fixtures (skip individual models if their file is missing) ────────────────

@pytest.fixture(scope="module")
def match_winner_model():
    path = MODELS_DIR / "match_winner_model.json"
    if not path.exists():
        pytest.skip("match_winner_model.json not found")
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m


@pytest.fixture(scope="module")
def toss_impact_model():
    path = MODELS_DIR / "toss_impact_model.json"
    if not path.exists():
        pytest.skip("toss_impact_model.json not found")
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m


@pytest.fixture(scope="module")
def batting_first_model():
    path = MODELS_DIR / "batting_first_model.json"
    if not path.exists():
        pytest.skip("batting_first_model.json not found")
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m


@pytest.fixture(scope="module")
def player_performance_model():
    path = MODELS_DIR / "player_performance_model.json"
    if not path.exists():
        pytest.skip("player_performance_model.json not found")
    m = xgb.XGBRegressor()
    m.load_model(str(path))
    return m


@pytest.fixture(scope="module")
def venue_advantage_model():
    path = MODELS_DIR / "venue_advantage_model.json"
    if not path.exists():
        pytest.skip("venue_advantage_model.json not found")
    m = xgb.XGBRegressor()
    m.load_model(str(path))
    return m


# ── Model file existence ──────────────────────────────────────────────────────

class TestModelFiles:
    """Basic file-presence checks — these run even when models are absent
    and report clearly what is missing."""

    def test_models_directory_exists(self):
        assert MODELS_DIR.exists(), "models/ directory is missing"

    @pytest.mark.parametrize("fname", _ALL_MODEL_FILES)
    def test_model_file_exists(self, fname):
        path = MODELS_DIR / fname
        if not path.exists():
            pytest.skip(f"{fname} not yet trained")
        assert path.exists()

    @pytest.mark.parametrize("fname", _ALL_MODEL_FILES)
    def test_model_file_non_empty(self, fname):
        path = MODELS_DIR / fname
        if not path.exists():
            pytest.skip(f"{fname} not yet trained")
        assert path.stat().st_size > 1_000, f"Model file suspiciously small: {fname}"


# ── Match winner model ────────────────────────────────────────────────────────
# Training features (8): t1_hist_win_rate, t2_hist_win_rate, h2h_t1_rate,
#                        toss_winner_is_t1, toss_decision_bat,
#                        venue_bf_rate, strength_diff, season_norm

class TestMatchWinnerModel:
    def test_strong_team_favoured(self, match_winner_model):
        """MI (0.62) vs PWI (0.35) — MI should be favoured."""
        # strength_diff for MI vs PWI = 0.62-0.35 = 0.27
        X_mi_wins  = np.array([[0.62, 0.35, 0.70, 1, 0, 0.47,  0.27, 0.9]])
        X_pwi_wins = np.array([[0.35, 0.62, 0.30, 0, 0, 0.47, -0.27, 0.9]])
        p_mi  = match_winner_model.predict_proba(X_mi_wins)[0][1]
        p_pwi = match_winner_model.predict_proba(X_pwi_wins)[0][1]
        assert p_mi > p_pwi, "Strong team (MI) should beat weak team (PWI) more often"

    def test_output_is_binary(self, match_winner_model):
        X = np.array([[0.55, 0.50, 0.55, 1, 1, 0.50, 0.05, 0.7]])
        pred = match_winner_model.predict(X)
        assert pred[0] in (0, 1)

    def test_probabilities_sum_to_one(self, match_winner_model):
        X = np.array([[0.55, 0.50, 0.55, 1, 1, 0.50, 0.05, 0.7]])
        probs = match_winner_model.predict_proba(X)[0]
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_output_probability_in_range(self, match_winner_model):
        rng = np.random.default_rng(42)
        X = rng.random((10, 8))
        probs = match_winner_model.predict_proba(X)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_batch_prediction(self, match_winner_model):
        rng = np.random.default_rng(42)
        X = rng.random((50, 8))
        preds = match_winner_model.predict(X)
        assert len(preds) == 50
        assert set(preds).issubset({0, 1})

    def test_feature_count_accepted(self, match_winner_model):
        """Model must accept exactly 8 features without error."""
        X = np.zeros((1, 8))
        match_winner_model.predict_proba(X)  # should not raise


# ── Toss impact model ─────────────────────────────────────────────────────────
# Training features (5): toss_decision_bat, venue_bf_rate, season_norm,
#                        t1_hist_win_rate, t2_hist_win_rate

class TestTossImpactModel:
    def test_loads_and_predicts(self, toss_impact_model):
        X = np.array([[1, 0.55, 0.8, 0.55, 0.50]])
        pred = toss_impact_model.predict(X)
        assert pred[0] in (0, 1)

    def test_probabilities_valid(self, toss_impact_model):
        X = np.array([[1, 0.55, 0.8, 0.55, 0.50]])
        probs = toss_impact_model.predict_proba(X)[0]
        assert abs(probs.sum() - 1.0) < 1e-5
        assert (probs >= 0).all()

    def test_batch_predictions(self, toss_impact_model):
        rng = np.random.default_rng(42)
        X = rng.random((20, 5))
        preds = toss_impact_model.predict(X)
        assert len(preds) == 20

    def test_feature_count_accepted(self, toss_impact_model):
        """Model must accept exactly 5 features without error."""
        X = np.zeros((1, 5))
        toss_impact_model.predict_proba(X)  # should not raise


# ── Batting first model ───────────────────────────────────────────────────────
# Training features (6): venue_bf_rate, first_innings_score, batting_first_is_t1,
#                        t1_hist_win_rate, t2_hist_win_rate, season_norm

class TestBattingFirstModel:
    def test_high_score_batting_first(self, batting_first_model):
        """200 runs at bat-first-friendly venue should beat 130."""
        X_high = np.array([[0.55, 200, 1, 0.55, 0.48, 0.8]])
        X_low  = np.array([[0.44, 130, 1, 0.55, 0.48, 0.8]])
        p_high = batting_first_model.predict_proba(X_high)[0][1]
        p_low  = batting_first_model.predict_proba(X_low)[0][1]
        assert p_high > p_low, "High first-innings score should favour batting-first team"

    def test_output_binary(self, batting_first_model):
        X = np.array([[0.50, 160, 1, 0.52, 0.50, 0.7]])
        pred = batting_first_model.predict(X)
        assert pred[0] in (0, 1)

    def test_batch_predictions(self, batting_first_model):
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.uniform(0.43, 0.56, 30),
            rng.integers(130, 210, 30),
            rng.integers(0, 2, 30),
            rng.uniform(0.40, 0.65, 30),
            rng.uniform(0.40, 0.65, 30),
            rng.uniform(0, 1, 30),
        ])
        preds = batting_first_model.predict(X)
        assert len(preds) == 30

    def test_feature_count_accepted(self, batting_first_model):
        """Model must accept exactly 6 features without error."""
        X = np.zeros((1, 6))
        batting_first_model.predict(X)  # should not raise


# ── Player performance model ──────────────────────────────────────────────────
# Training features (6): strength_rating, venue_bf_rate, role_enc,
#                        batting_avg, strike_rate, bowling_economy

class TestPlayerPerformanceModel:
    def test_predicts_score(self, player_performance_model):
        """Virat Kohli-like batsman at flat venue."""
        X = np.array([[0.58, 0.53, 0, 37.8, 130.1, 0.0]])
        score = player_performance_model.predict(X)[0]
        assert 15 < score < 95, f"Score out of realistic range: {score}"

    def test_strong_team_outperforms_weak(self, player_performance_model):
        X_strong = np.array([[0.62, 0.50, 0, 35.0, 138.0, 0.0]])
        X_weak   = np.array([[0.35, 0.50, 0, 18.0, 110.0, 0.0]])
        s_strong = player_performance_model.predict(X_strong)[0]
        s_weak   = player_performance_model.predict(X_weak)[0]
        assert s_strong > s_weak

    def test_batch_predictions(self, player_performance_model):
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.uniform(0.35, 0.62, 20),
            rng.uniform(0.43, 0.56, 20),
            rng.integers(0, 3, 20),
            rng.uniform(0, 50, 20),
            rng.uniform(0, 180, 20),
            rng.uniform(0, 12, 20),
        ])
        scores = player_performance_model.predict(X)
        assert len(scores) == 20
        assert (scores > 0).all()

    def test_bowler_features(self, player_performance_model):
        """Bowler has 0 batting stats — model should still predict."""
        X = np.array([[0.52, 0.48, 1, 5.0, 0.0, 7.5]])  # role_enc=1 (bowler)
        score = player_performance_model.predict(X)[0]
        assert isinstance(float(score), float)

    def test_feature_count_accepted(self, player_performance_model):
        """Model must accept exactly 6 features without error."""
        X = np.zeros((1, 6))
        player_performance_model.predict(X)  # should not raise


# ── Venue advantage model ─────────────────────────────────────────────────────
# Training features (4): strength, global_win_rate, venue_bf_rate, avg_score

class TestVenueAdvantageModel:
    def test_predicts_boost(self, venue_advantage_model):
        X = np.array([[0.62, 0.58, 0.47, 0.84]])  # MI at Wankhede (avg_score normalised)
        boost = venue_advantage_model.predict(X)[0]
        assert isinstance(float(boost), float)

    def test_batch_predictions(self, venue_advantage_model):
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.uniform(0.35, 0.62, 15),
            rng.uniform(0.35, 0.62, 15),
            rng.uniform(0.43, 0.56, 15),
            rng.uniform(0.75, 0.90, 15),
        ])
        boosts = venue_advantage_model.predict(X)
        assert len(boosts) == 15

    def test_home_venue_positive_boost(self, venue_advantage_model):
        """Strong home team should have higher boost than weak away team."""
        X_home = np.array([[0.62, 0.58, 0.50, 0.84]])
        X_away = np.array([[0.35, 0.35, 0.43, 0.78]])
        boost_home = venue_advantage_model.predict(X_home)[0]
        boost_away = venue_advantage_model.predict(X_away)[0]
        assert boost_home >= boost_away

    def test_feature_count_accepted(self, venue_advantage_model):
        """Model must accept exactly 4 features without error."""
        X = np.zeros((1, 4))
        venue_advantage_model.predict(X)  # should not raise


# ── Predictor singleton integration ──────────────────────────────────────────

class TestPredictorIntegration:
    """Test the IPLPredictor singleton with the loaded models."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from src.predictor import predictor
        predictor.load_models()
        self.predictor = predictor

    def test_predictor_load_models_returns_bool(self):
        result = self.predictor.load_models()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not _ALL_MODELS_AVAILABLE, reason="Models not trained yet")
    def test_predictor_is_loaded_true_when_models_present(self):
        assert self.predictor.is_loaded is True

    def test_predict_match_winner_fallback(self):
        """predict_match_winner works without models loaded (heuristic fallback)."""
        result = self.predictor.predict_match_winner(
            "MI", "CSK", "Wankhede Stadium, Mumbai", "MI", "bat"
        )
        assert result["predicted_winner"] in ("MI", "CSK")
        assert 0.0 < result["win_probability"] <= 1.0

    def test_predict_toss_impact_fallback(self):
        result = self.predictor.predict_toss_impact(
            "Wankhede Stadium, Mumbai", "bat"
        )
        assert "batting_first_win_probability" in result
        total = result["batting_first_win_probability"] + result["chasing_win_probability"]
        assert abs(total - 1.0) < 0.01

    def test_predict_player_performance_fallback(self):
        result = self.predictor.predict_player_performance(
            "Rohit Sharma", "MI", "Wankhede Stadium, Mumbai", "batsman"
        )
        assert 0 < result["predicted_performance_score"] < 100
        assert result["performance_tier"] in ("Elite", "Strong", "Average", "Below average")


# ── Training report ───────────────────────────────────────────────────────────

class TestTrainingReport:
    REPORT_PATH = DATA_DIR / "training_report.json"

    def test_report_exists(self):
        if not self.REPORT_PATH.exists():
            pytest.skip("training_report.json not found — run scripts/train_models.py")
        assert self.REPORT_PATH.exists()

    def test_report_has_all_models(self):
        if not self.REPORT_PATH.exists():
            pytest.skip("training_report.json not found")
        import json
        with open(self.REPORT_PATH) as f:
            report = json.load(f)
        expected = {"match_winner", "toss_impact", "batting_first",
                    "player_performance", "venue_advantage"}
        assert expected.issubset(set(report.keys()))

    def test_classifier_accuracy_above_threshold(self):
        if not self.REPORT_PATH.exists():
            pytest.skip("training_report.json not found")
        import json
        with open(self.REPORT_PATH) as f:
            report = json.load(f)
        for name in ("match_winner", "toss_impact", "batting_first"):
            acc = report[name]["test_accuracy"]
            assert acc >= 0.50, f"{name} accuracy {acc:.3f} is below 50% random baseline"

    def test_regressor_r2_reported(self):
        if not self.REPORT_PATH.exists():
            pytest.skip("training_report.json not found")
        import json
        with open(self.REPORT_PATH) as f:
            report = json.load(f)
        for name in ("player_performance", "venue_advantage"):
            assert "r2" in report[name], f"Missing r2 in {name} report"
            assert "mae" in report[name], f"Missing mae in {name} report"
