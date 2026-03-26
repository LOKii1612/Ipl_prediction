"""
Train 5 XGBoost models on the IPL DuckDB dataset and save them to models/.

Models trained
──────────────
1. match_winner_model.json    – binary: team1 wins? (XGBClassifier)
2. toss_impact_model.json     – binary: toss winner wins? (XGBClassifier)
3. batting_first_model.json   – binary: batting-first team wins? (XGBClassifier)
4. player_performance_model.json – regression: player performance score (XGBRegressor)
5. venue_advantage_model.json – regression: batting-first win probability boost (XGBRegressor)
"""

import json
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

SEED = 42
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "ipl_matches.duckdb"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORT_PATH = ROOT / "data" / "training_report.json"

CLASSIFIER_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=SEED,
    verbosity=0,
)

REGRESSOR_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbosity=0,
)


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_matches() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("SELECT * FROM matches").df()
    conn.close()
    return df


def load_performances() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("SELECT * FROM player_performances").df()
    conn.close()
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

TEAM_STRENGTH = {
    "MI": 0.62, "CSK": 0.61, "GT": 0.58, "KKR": 0.53,
    "SRH": 0.52, "RR": 0.51, "LSG": 0.49, "DC": 0.48,
    "RCB": 0.47, "PBKS": 0.46, "RPS": 0.46, "KXIP": 0.44,
    "DD": 0.43, "PWI": 0.35,
}

VENUE_BF = {
    "Wankhede Stadium, Mumbai": 0.47,
    "MA Chidambaram Stadium, Chennai": 0.44,
    "Eden Gardens, Kolkata": 0.50,
    "M Chinnaswamy Stadium, Bangalore": 0.53,
    "Rajiv Gandhi International Cricket Stadium, Hyderabad": 0.48,
    "Arun Jaitley Stadium, Delhi": 0.49,
    "Punjab Cricket Association Stadium, Mohali": 0.53,
    "Sawai Mansingh Stadium, Jaipur": 0.46,
    "Narendra Modi Stadium, Ahmedabad": 0.55,
    "BRSABV Ekana Cricket Stadium, Lucknow": 0.48,
}


def enrich_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-computed feature columns to matches dataframe."""
    # Team strength
    df["t1_strength"] = df["team1"].map(TEAM_STRENGTH).fillna(0.48)
    df["t2_strength"] = df["team2"].map(TEAM_STRENGTH).fillna(0.48)
    df["strength_diff"] = df["t1_strength"] - df["t2_strength"]

    # Historical win rates per team (computed from the full dataset)
    team_wins = {}
    all_teams = set(df["team1"].tolist() + df["team2"].tolist())
    for t in all_teams:
        played = len(df[(df["team1"] == t) | (df["team2"] == t)])
        won = len(df[df["winner"] == t])
        team_wins[t] = won / played if played > 0 else 0.48
    df["t1_hist_win_rate"] = df["team1"].map(team_wins).fillna(0.48)
    df["t2_hist_win_rate"] = df["team2"].map(team_wins).fillna(0.48)

    # Toss features
    df["toss_winner_is_t1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["toss_decision_bat"] = (df["toss_decision"] == "bat").astype(int)
    df["batting_first_is_t1"] = (df["batting_first"] == df["team1"]).astype(int)

    # Venue features
    df["venue_bf_rate"] = df["venue"].map(VENUE_BF).fillna(0.50)
    df["score_diff"] = df["first_innings_score"] - df["second_innings_score"]
    df["season_norm"] = (df["season"] - 2008) / 16.0  # normalize to 0-1

    # Head-to-head win ratio (computed from dataset)
    h2h_t1 = {}
    for _, row in df.iterrows():
        pair = (row["team1"], row["team2"])
        h2h_t1.setdefault(pair, {"wins": 0, "total": 0})
        h2h_t1[pair]["total"] += 1
        if row["winner"] == row["team1"]:
            h2h_t1[pair]["wins"] += 1
    df["h2h_t1_rate"] = df.apply(
        lambda r: h2h_t1.get((r["team1"], r["team2"]), {}).get("wins", 0) /
                  max(1, h2h_t1.get((r["team1"], r["team2"]), {}).get("total", 1)),
        axis=1
    )

    return df


# ── Model 1: Match Winner ─────────────────────────────────────────────────────

def train_match_winner(df: pd.DataFrame) -> dict:
    print("\n[1/5] Training match_winner_model …")
    features = [
        "t1_hist_win_rate", "t2_hist_win_rate", "h2h_t1_rate",
        "toss_winner_is_t1", "toss_decision_bat",
        "venue_bf_rate", "strength_diff", "season_norm",
    ]
    X = df[features].values
    y = df["team1_win"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = xgb.XGBClassifier(**CLASSIFIER_PARAMS)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    model.save_model(str(MODELS_DIR / "match_winner_model.json"))
    print(f"   Test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   CV accuracy   : {cv:.4f}  ({cv*100:.1f}%)")

    return {"model": "match_winner", "test_accuracy": round(acc, 4),
            "cv_accuracy": round(cv, 4), "features": features}


# ── Model 2: Toss Impact ──────────────────────────────────────────────────────

def train_toss_impact(df: pd.DataFrame) -> dict:
    print("\n[2/5] Training toss_impact_model …")
    features = [
        "toss_decision_bat", "venue_bf_rate", "season_norm",
        "t1_hist_win_rate", "t2_hist_win_rate",
    ]
    X = df[features].values
    y = df["toss_winner_won"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = xgb.XGBClassifier(**CLASSIFIER_PARAMS)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    model.save_model(str(MODELS_DIR / "toss_impact_model.json"))
    print(f"   Test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   CV accuracy   : {cv:.4f}  ({cv*100:.1f}%)")

    return {"model": "toss_impact", "test_accuracy": round(acc, 4),
            "cv_accuracy": round(cv, 4), "features": features}


# ── Model 3: Batting First ────────────────────────────────────────────────────

def train_batting_first(df: pd.DataFrame) -> dict:
    print("\n[3/5] Training batting_first_model …")
    features = [
        "venue_bf_rate", "first_innings_score", "batting_first_is_t1",
        "t1_hist_win_rate", "t2_hist_win_rate", "season_norm",
    ]
    df["batting_first_won"] = (df["result"] == "batting_first").astype(int)
    X = df[features].values
    y = df["batting_first_won"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = xgb.XGBClassifier(**CLASSIFIER_PARAMS)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    model.save_model(str(MODELS_DIR / "batting_first_model.json"))
    print(f"   Test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   CV accuracy   : {cv:.4f}  ({cv*100:.1f}%)")

    return {"model": "batting_first", "test_accuracy": round(acc, 4),
            "cv_accuracy": round(cv, 4), "features": features}


# ── Model 4: Player Performance ───────────────────────────────────────────────

def train_player_performance(perf_df: pd.DataFrame) -> dict:
    print("\n[4/5] Training player_performance_model …")
    role_enc = {"batsman": 0, "bowler": 1, "allrounder": 2}
    perf_df["role_enc"] = perf_df["role"].map(role_enc).fillna(0)

    features = [
        "strength_rating", "venue_bf_rate", "role_enc",
        "batting_avg", "strike_rate", "bowling_economy",
    ]
    X = perf_df[features].fillna(0).values
    y = perf_df["performance_score"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = xgb.XGBRegressor(**REGRESSOR_PARAMS)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    model.save_model(str(MODELS_DIR / "player_performance_model.json"))
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")

    return {"model": "player_performance", "mae": round(mae, 4),
            "r2": round(r2, 4), "features": features}


# ── Model 5: Venue Advantage ──────────────────────────────────────────────────

def train_venue_advantage(df: pd.DataFrame) -> dict:
    print("\n[5/5] Training venue_advantage_model …")

    # Aggregate actual home-team win rates per venue
    venue_rows = []
    all_venues = df["venue"].unique()
    all_teams = set(df["team1"].tolist() + df["team2"].tolist())
    team_wins_global = {}
    for t in all_teams:
        pl = len(df[(df["team1"] == t) | (df["team2"] == t)])
        wo = len(df[df["winner"] == t])
        team_wins_global[t] = wo / pl if pl > 0 else 0.48

    for _, row in df.iterrows():
        venue = row["venue"]
        venue_matches = df[df["venue"] == venue]
        vbf = VENUE_BF.get(venue, 0.50)
        v_avg = venue_matches["first_innings_score"].mean() if len(venue_matches) else 162.0

        for team in (row["team1"], row["team2"]):
            t_at_v = venue_matches[(venue_matches["team1"] == team) | (venue_matches["team2"] == team)]
            t_wins_at_v = len(t_at_v[t_at_v["winner"] == team])
            t_played_at_v = len(t_at_v)
            home_win_rate = t_wins_at_v / t_played_at_v if t_played_at_v > 0 else team_wins_global.get(team, 0.48)
            global_win_rate = team_wins_global.get(team, 0.48)
            home_boost = home_win_rate - global_win_rate  # target: venue advantage boost

            venue_rows.append({
                "strength": TEAM_STRENGTH.get(team, 0.48),
                "global_win_rate": global_win_rate,
                "venue_bf_rate": vbf,
                "avg_score": v_avg / 200.0,  # normalised
                "home_boost": home_boost,
            })

    vdf = pd.DataFrame(venue_rows).dropna()
    features = ["strength", "global_win_rate", "venue_bf_rate", "avg_score"]
    X = vdf[features].values
    y = vdf["home_boost"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = xgb.XGBRegressor(**REGRESSOR_PARAMS)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    model.save_model(str(MODELS_DIR / "venue_advantage_model.json"))
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")

    return {"model": "venue_advantage", "mae": round(mae, 4),
            "r2": round(r2, 4), "features": features}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data from DuckDB …")
    matches_df = load_matches()
    perf_df    = load_performances()
    print(f"  Matches loaded      : {len(matches_df)}")
    print(f"  Performances loaded : {len(perf_df)}")

    matches_df = enrich_matches(matches_df)

    report = {}
    report["match_winner"]       = train_match_winner(matches_df)
    report["toss_impact"]        = train_toss_impact(matches_df)
    report["batting_first"]      = train_batting_first(matches_df)
    report["player_performance"] = train_player_performance(perf_df)
    report["venue_advantage"]    = train_venue_advantage(matches_df)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, metrics in report.items():
        if "test_accuracy" in metrics:
            print(f"  {name:<30} accuracy={metrics['test_accuracy']*100:.1f}%  cv={metrics['cv_accuracy']*100:.1f}%")
        else:
            print(f"  {name:<30} MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Report saved to: {REPORT_PATH}")
