"""
Generate synthetic IPL T20 match data mimicking real patterns
and load it into DuckDB at data/ipl_matches.duckdb.

Synthetic data characteristics:
  - 14 IPL teams with historically-calibrated strength ratings
  - 22 venues with realistic batting-first win rates and average scores
  - 1 200 matches spanning IPL seasons 2008–2024
  - Toss decisions correlated with venue type (pitch/conditions)
  - Player performance data for 70 key players across all teams
  - Feature-rich enough for XGBoost training (signal in every column)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "ipl_matches.duckdb"

# ── Team configuration ────────────────────────────────────────────────────────

TEAMS = {
    "MI":   {"strength": 0.62, "home": "Wankhede Stadium, Mumbai"},
    "CSK":  {"strength": 0.61, "home": "MA Chidambaram Stadium, Chennai"},
    "GT":   {"strength": 0.58, "home": "Narendra Modi Stadium, Ahmedabad"},
    "KKR":  {"strength": 0.53, "home": "Eden Gardens, Kolkata"},
    "SRH":  {"strength": 0.52, "home": "Rajiv Gandhi International Cricket Stadium, Hyderabad"},
    "RR":   {"strength": 0.51, "home": "Sawai Mansingh Stadium, Jaipur"},
    "LSG":  {"strength": 0.49, "home": "BRSABV Ekana Cricket Stadium, Lucknow"},
    "DC":   {"strength": 0.48, "home": "Arun Jaitley Stadium, Delhi"},
    "RCB":  {"strength": 0.47, "home": "M Chinnaswamy Stadium, Bangalore"},
    "PBKS": {"strength": 0.46, "home": "Punjab Cricket Association Stadium, Mohali"},
    "RPS":  {"strength": 0.46, "home": "Maharashtra Cricket Association Stadium, Pune"},
    "KXIP": {"strength": 0.44, "home": "Punjab Cricket Association Stadium, Mohali"},
    "DD":   {"strength": 0.43, "home": "Feroz Shah Kotla, Delhi"},
    "PWI":  {"strength": 0.35, "home": "Subrata Roy Sahara Stadium, Pune"},
}

# ── Venue configuration ───────────────────────────────────────────────────────

VENUES = [
    {"name": "Wankhede Stadium, Mumbai",                             "city": "Mumbai",             "bf": 0.47, "avg": 168, "home": "MI"},
    {"name": "MA Chidambaram Stadium, Chennai",                      "city": "Chennai",            "bf": 0.44, "avg": 158, "home": "CSK"},
    {"name": "Eden Gardens, Kolkata",                                "city": "Kolkata",            "bf": 0.50, "avg": 162, "home": "KKR"},
    {"name": "M Chinnaswamy Stadium, Bangalore",                     "city": "Bangalore",          "bf": 0.53, "avg": 174, "home": "RCB"},
    {"name": "Rajiv Gandhi International Cricket Stadium, Hyderabad","city": "Hyderabad",          "bf": 0.48, "avg": 161, "home": "SRH"},
    {"name": "Arun Jaitley Stadium, Delhi",                          "city": "Delhi",              "bf": 0.49, "avg": 163, "home": "DC"},
    {"name": "Punjab Cricket Association Stadium, Mohali",           "city": "Mohali",             "bf": 0.53, "avg": 166, "home": "PBKS"},
    {"name": "Sawai Mansingh Stadium, Jaipur",                       "city": "Jaipur",             "bf": 0.46, "avg": 160, "home": "RR"},
    {"name": "Narendra Modi Stadium, Ahmedabad",                     "city": "Ahmedabad",          "bf": 0.55, "avg": 170, "home": "GT"},
    {"name": "BRSABV Ekana Cricket Stadium, Lucknow",                "city": "Lucknow",            "bf": 0.48, "avg": 162, "home": "LSG"},
    {"name": "Maharashtra Cricket Association Stadium, Pune",        "city": "Pune",               "bf": 0.51, "avg": 165, "home": "RPS"},
    {"name": "Brabourne Stadium, Mumbai",                            "city": "Mumbai",             "bf": 0.50, "avg": 164, "home": "MI"},
    {"name": "DY Patil Stadium, Mumbai",                             "city": "Mumbai",             "bf": 0.52, "avg": 167, "home": "MI"},
    {"name": "Subrata Roy Sahara Stadium, Pune",                     "city": "Pune",               "bf": 0.47, "avg": 158, "home": "PWI"},
    {"name": "Feroz Shah Kotla, Delhi",                              "city": "Delhi",              "bf": 0.49, "avg": 161, "home": "DD"},
    {"name": "Sharjah Cricket Stadium, Sharjah",                     "city": "Sharjah",            "bf": 0.43, "avg": 155, "home": None},
    {"name": "Dubai International Cricket Stadium, Dubai",           "city": "Dubai",              "bf": 0.44, "avg": 157, "home": None},
    {"name": "Sheikh Zayed Stadium, Abu Dhabi",                      "city": "Abu Dhabi",          "bf": 0.46, "avg": 156, "home": None},
    {"name": "Holkar Cricket Stadium, Indore",                       "city": "Indore",             "bf": 0.54, "avg": 169, "home": None},
    {"name": "JSCA International Stadium Complex, Ranchi",           "city": "Ranchi",             "bf": 0.48, "avg": 160, "home": None},
    {"name": "Greenfield International Stadium, Thiruvananthapuram", "city": "Thiruvananthapuram", "bf": 0.49, "avg": 161, "home": None},
    {"name": "Himachal Pradesh Cricket Association Stadium, Dharamsala", "city": "Dharamsala",    "bf": 0.52, "avg": 163, "home": None},
]

# ── Player configuration ──────────────────────────────────────────────────────

PLAYERS = {
    "MI":   [("Rohit Sharma",      "batsman",    31.2, 131.5, None),
             ("Suryakumar Yadav",  "batsman",    33.5, 157.2, None),
             ("Jasprit Bumrah",    "bowler",     None, None,  6.8),
             ("Kieron Pollard",    "allrounder", 26.8, 146.3, 8.9),
             ("Ishan Kishan",      "batsman",    28.4, 136.1, None)],
    "CSK":  [("MS Dhoni",          "batsman",    38.1, 134.5, None),
             ("Ruturaj Gaikwad",   "batsman",    35.8, 133.7, None),
             ("Ravindra Jadeja",   "allrounder", 22.1, 124.8, 7.2),
             ("Deepak Chahar",     "bowler",     None, None,  7.5),
             ("Ambati Rayudu",     "batsman",    26.5, 128.3, None)],
    "RCB":  [("Virat Kohli",       "batsman",    37.8, 130.1, None),
             ("AB de Villiers",    "batsman",    40.2, 158.9, None),
             ("Glenn Maxwell",     "allrounder", 29.6, 154.7, 8.5),
             ("Mohammed Siraj",    "bowler",     None, None,  8.1),
             ("Faf du Plessis",    "batsman",    32.4, 134.2, None)],
    "KKR":  [("Andre Russell",     "allrounder", 29.1, 177.8, 9.0),
             ("Shreyas Iyer",      "batsman",    30.5, 127.6, None),
             ("Sunil Narine",      "allrounder", 18.3, 155.2, 6.7),
             ("Pat Cummins",       "bowler",     None, None,  8.4),
             ("Venkatesh Iyer",    "allrounder", 26.8, 141.3, 8.6)],
    "SRH":  [("David Warner",      "batsman",    41.4, 139.8, None),
             ("Kane Williamson",   "batsman",    38.6, 124.5, None),
             ("Bhuvneshwar Kumar", "bowler",     None, None,  7.3),
             ("Rashid Khan",       "bowler",     None, None,  6.3),
             ("Jonny Bairstow",    "batsman",    31.2, 136.7, None)],
    "DC":   [("Rishabh Pant",      "batsman",    34.3, 148.2, None),
             ("Shikhar Dhawan",    "batsman",    33.6, 127.9, None),
             ("Axar Patel",        "allrounder", 19.4, 127.5, 6.8),
             ("Kagiso Rabada",     "bowler",     None, None,  8.3),
             ("Prithvi Shaw",      "batsman",    28.8, 147.6, None)],
    "RR":   [("Sanju Samson",      "batsman",    32.6, 138.4, None),
             ("Jos Buttler",       "batsman",    45.2, 148.9, None),
             ("Yuzvendra Chahal",  "bowler",     None, None,  7.6),
             ("Ben Stokes",        "allrounder", 26.9, 130.2, 8.7),
             ("Trent Boult",       "bowler",     None, None,  8.0)],
    "GT":   [("Hardik Pandya",     "allrounder", 27.3, 144.5, 9.1),
             ("Shubman Gill",      "batsman",    37.8, 133.8, None),
             ("Mohammed Shami",    "bowler",     None, None,  7.8),
             ("David Miller",      "batsman",    34.2, 141.7, None),
             ("Rashid Khan",       "bowler",     None, None,  6.5)],
    "LSG":  [("KL Rahul",          "batsman",    47.8, 133.4, None),
             ("Quinton de Kock",   "batsman",    30.5, 137.2, None),
             ("Avesh Khan",        "bowler",     None, None,  8.6),
             ("Marcus Stoinis",    "allrounder", 25.4, 141.9, 8.9),
             ("Deepak Hooda",      "allrounder", 22.7, 139.4, 8.4)],
    "PBKS": [("Shikhar Dhawan",    "batsman",    31.8, 125.7, None),
             ("Liam Livingstone",  "allrounder", 27.1, 160.8, 8.8),
             ("Kagiso Rabada",     "bowler",     None, None,  8.0),
             ("Arshdeep Singh",    "bowler",     None, None,  7.9),
             ("Jonny Bairstow",    "batsman",    30.5, 138.2, None)],
    "KXIP": [("Chris Gayle",       "batsman",    44.2, 148.9, None),
             ("KL Rahul",          "batsman",    53.8, 138.8, None),
             ("Glenn Maxwell",     "allrounder", 21.5, 161.5, 9.0),
             ("Mohammed Shami",    "bowler",     None, None,  8.1),
             ("Nicholas Pooran",   "batsman",    27.9, 152.4, None)],
    "DD":   [("Shikhar Dhawan",    "batsman",    34.2, 128.3, None),
             ("Rishabh Pant",      "batsman",    28.7, 151.9, None),
             ("Amit Mishra",       "bowler",     None, None,  7.2),
             ("Chris Morris",      "allrounder", 18.6, 142.3, 8.3),
             ("Shreyas Iyer",      "batsman",    27.4, 125.9, None)],
    "RPS":  [("Steve Smith",       "batsman",    35.6, 119.4, None),
             ("Ben Stokes",        "allrounder", 24.8, 133.7, 9.2),
             ("MS Dhoni",          "batsman",    36.2, 130.1, None),
             ("Imran Tahir",       "bowler",     None, None,  7.1),
             ("Ajinkya Rahane",    "batsman",    29.4, 116.7, None)],
    "PWI":  [("Sourav Ganguly",    "batsman",    25.4, 123.6, None),
             ("Yuvraj Singh",      "allrounder", 32.1, 129.8, 8.8),
             ("Angelo Mathews",    "allrounder", 22.8, 115.3, 8.5),
             ("Ashok Dinda",       "bowler",     None, None,  8.9),
             ("Robin Uthappa",     "batsman",    27.6, 131.4, None)],
}


# ── Data generation ───────────────────────────────────────────────────────────

def _win_prob(t1: str, t2: str, venue: dict, batting_first: str) -> float:
    """Compute team1 win probability given context."""
    s1 = TEAMS[t1]["strength"]
    s2 = TEAMS[t2]["strength"]

    # Home advantage
    if venue["home"] == t1:
        s1 += 0.08
    elif venue["home"] == t2:
        s2 += 0.08

    # Batting-first advantage
    bf_rate = venue["bf"]
    bf_bonus = abs(bf_rate - 0.50) * 0.15
    if batting_first == t1:
        if bf_rate >= 0.50:
            s1 += bf_bonus
        else:
            s2 += bf_bonus
    else:
        if bf_rate >= 0.50:
            s2 += bf_bonus
        else:
            s1 += bf_bonus

    return s1 / (s1 + s2)


def generate_matches(n: int = 1200) -> pd.DataFrame:
    teams = list(TEAMS.keys())
    start = datetime(2008, 4, 18)
    rows = []

    for i in range(n):
        t1, t2 = random.sample(teams, 2)
        venue = random.choice(VENUES)

        # Toss
        toss_winner = random.choice([t1, t2])
        # Teams prefer to chase at slow surfaces (bf < 0.50), bat at flat ones
        prefer_bat_prob = 0.40 + (venue["bf"] - 0.44) * 2.0
        prefer_bat_prob = max(0.25, min(0.75, prefer_bat_prob))
        toss_decision = "bat" if random.random() < prefer_bat_prob else "field"
        batting_first = toss_winner if toss_decision == "bat" else (t2 if toss_winner == t1 else t1)

        p1 = _win_prob(t1, t2, venue, batting_first)
        t1_wins = random.random() < p1
        winner = t1 if t1_wins else t2

        avg = venue["avg"]
        if batting_first == winner:
            fi = int(np.clip(np.random.normal(avg + 8, 18), 90, 250))
            si = int(np.clip(np.random.normal(avg - 12, 20), 60, fi - 1))
            result_type = "runs"
            margin = fi - si
        else:
            fi = int(np.clip(np.random.normal(avg - 5, 15), 90, 235))
            si = int(np.clip(np.random.normal(avg + 5, 12), fi + 1, 250))
            result_type = "wickets"
            margin = random.randint(1, 8)

        match_date = start + timedelta(days=i * 2 + random.randint(-3, 3))

        rows.append({
            "match_id": i + 1,
            "date": match_date.strftime("%Y-%m-%d"),
            "season": match_date.year,
            "venue": venue["name"],
            "city": venue["city"],
            "team1": t1,
            "team2": t2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "batting_first": batting_first,
            "first_innings_score": fi,
            "second_innings_score": si,
            "winner": winner,
            "result_type": result_type,
            "match_margin": margin,
            "result": "batting_first" if batting_first == winner else "chasing",
            "team1_win": int(t1_wins),
            "toss_winner_won": int(toss_winner == winner),
        })

    return pd.DataFrame(rows)


def generate_players() -> pd.DataFrame:
    rows = []
    pid = 1
    for team, players in PLAYERS.items():
        for (name, role, b_avg, sr, eco) in players:
            rows.append({
                "player_id": pid,
                "player_name": name,
                "team": team,
                "role": role,
                "batting_avg": b_avg if b_avg is not None else 0.0,
                "strike_rate": sr if sr is not None else 0.0,
                "bowling_economy": eco if eco is not None else 0.0,
                "matches_played": random.randint(40, 200),
                "strength_rating": TEAMS[team]["strength"],
            })
            pid += 1
    return pd.DataFrame(rows)


def generate_player_match_performances(matches_df: pd.DataFrame,
                                       players_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pid = 1
    for _, match in matches_df.sample(min(len(matches_df), 400), random_state=SEED).iterrows():
        for team in (match["team1"], match["team2"]):
            team_players = players_df[players_df["team"] == team]
            for _, player in team_players.iterrows():
                role = player["role"]
                if role == "batsman":
                    score = max(0, int(np.random.normal(player["batting_avg"], 15)))
                    balls = max(1, int(score / (player["strike_rate"] / 100 + 0.01)))
                    wickets_taken, runs_given = 0, 0
                elif role == "bowler":
                    score, balls = 0, 0
                    wickets_taken = int(np.clip(np.random.poisson(1.2), 0, 5))
                    runs_given = int(np.random.normal(player["bowling_economy"] * 4, 5))
                else:  # allrounder
                    score = max(0, int(np.random.normal(player["batting_avg"] * 0.8, 12)))
                    balls = max(1, int(score / (player["strike_rate"] / 100 + 0.01)))
                    wickets_taken = int(np.clip(np.random.poisson(0.8), 0, 3))
                    runs_given = int(np.random.normal(player["bowling_economy"] * 3.5, 4))

                # composite performance score (0-100)
                bat_contribution = min(40, score * 0.5)
                bowl_contribution = min(30, wickets_taken * 10 - max(0, runs_given - 25) * 0.3)
                perf_score = max(5, bat_contribution + bowl_contribution + 15)

                rows.append({
                    "perf_id": pid,
                    "match_id": match["match_id"],
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "team": team,
                    "role": role,
                    "runs_scored": score,
                    "balls_faced": balls,
                    "wickets_taken": wickets_taken,
                    "runs_given": runs_given,
                    "performance_score": round(perf_score, 2),
                    "batting_avg": player["batting_avg"],
                    "strike_rate": player["strike_rate"],
                    "bowling_economy": player["bowling_economy"],
                    "strength_rating": player["strength_rating"],
                    "venue": match["venue"],
                    "venue_bf_rate": next(
                        (v["bf"] for v in VENUES if v["name"] == match["venue"]), 0.50
                    ),
                })
                pid += 1

    return pd.DataFrame(rows)


# ── DuckDB loader ─────────────────────────────────────────────────────────────

def load_to_duckdb(matches: pd.DataFrame, players: pd.DataFrame,
                   performances: pd.DataFrame) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = duckdb.connect(str(DB_PATH))

    conn.execute("""
        CREATE TABLE matches AS SELECT * FROM matches
    """)
    conn.execute("""
        CREATE TABLE players AS SELECT * FROM players
    """)
    conn.execute("""
        CREATE TABLE player_performances AS SELECT * FROM performances
    """)

    # Venue summary view
    conn.execute("""
        CREATE VIEW venue_stats AS
        SELECT
            venue,
            city,
            COUNT(*) AS total_matches,
            ROUND(AVG(first_innings_score), 1) AS avg_first_innings,
            ROUND(AVG(second_innings_score), 1) AS avg_second_innings,
            ROUND(SUM(CASE WHEN result = 'batting_first' THEN 1 ELSE 0 END)
                  * 100.0 / COUNT(*), 1) AS batting_first_win_pct
        FROM matches
        GROUP BY venue, city
    """)

    # H2H view
    conn.execute("""
        CREATE VIEW h2h_stats AS
        SELECT
            LEAST(team1, team2)   AS team_a,
            GREATEST(team1, team2) AS team_b,
            COUNT(*) AS total,
            SUM(CASE WHEN winner = LEAST(team1, team2) THEN 1 ELSE 0 END) AS team_a_wins
        FROM matches
        GROUP BY LEAST(team1, team2), GREATEST(team1, team2)
    """)

    n_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    n_players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    n_perfs   = conn.execute("SELECT COUNT(*) FROM player_performances").fetchone()[0]

    conn.close()
    print(f"DuckDB loaded: {n_matches} matches | {n_players} players | {n_perfs} performances")
    print(f"Database path: {DB_PATH}  size: {DB_PATH.stat().st_size / 1024:.1f} KB")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 1 200 synthetic IPL matches …")
    matches = generate_matches(1200)

    print("Generating player roster …")
    players = generate_players()

    print("Generating player match performances …")
    performances = generate_player_match_performances(matches, players)

    print("Loading into DuckDB …")
    load_to_duckdb(matches, players, performances)

    print("\nSample matches:")
    print(matches[["team1", "team2", "venue", "winner", "first_innings_score"]].head(5).to_string(index=False))
    print("\nTeam win rates:")
    wr = matches.groupby("winner").size() / len(matches) * 100
    print(wr.sort_values(ascending=False).to_string())
    print("\nData generation complete.")
