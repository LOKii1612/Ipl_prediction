"""DuckDB query layer for IPL match data — singleton connection."""

import os
import logging
import threading
import duckdb
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data")) / "ipl_matches.duckdb"

_lock = threading.Lock()
_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a singleton read-only DuckDB connection (thread-safe)."""
    global _conn
    with _lock:
        if _conn is None:
            if not DB_PATH.exists():
                raise FileNotFoundError(f"Database not found: {DB_PATH}")
            _conn = duckdb.connect(str(DB_PATH), read_only=True)
            logger.info("DuckDB connection opened: %s", DB_PATH)
        return _conn


def close_connection() -> None:
    """Close the singleton connection (call on shutdown)."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
            logger.info("DuckDB connection closed")


def get_venue_stats(venue: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT
                venue,
                COUNT(*) AS total_matches,
                ROUND(AVG(first_innings_score), 1)  AS avg_first_innings,
                ROUND(AVG(second_innings_score), 1) AS avg_second_innings,
                ROUND(SUM(CASE WHEN result = 'batting_first' THEN 1 ELSE 0 END)
                      * 100.0 / COUNT(*), 1)        AS batting_first_win_pct,
                ROUND(AVG(match_margin), 1)          AS avg_margin
            FROM matches
            WHERE LOWER(venue) LIKE LOWER(?)
            GROUP BY venue
            LIMIT 1
        """, [f"%{venue}%"]).fetchone()

        if not row:
            return {}
        return {
            "venue": row[0],
            "total_matches": row[1],
            "avg_first_innings_score": row[2],
            "avg_second_innings_score": row[3],
            "batting_first_win_pct": row[4],
            "avg_margin": row[5],
        }
    except Exception as e:
        logger.error("get_venue_stats failed: %s", e)
        return {}


def get_head_to_head(team1: str, team2: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) AS t1_wins,
                SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) AS t2_wins,
                ROUND(AVG(CASE WHEN batting_first = ? THEN first_innings_score
                               WHEN batting_first = ? THEN second_innings_score END), 1) AS t1_avg,
                ROUND(AVG(CASE WHEN batting_first = ? THEN first_innings_score
                               WHEN batting_first = ? THEN second_innings_score END), 1) AS t2_avg
            FROM matches
            WHERE (team1 = ? AND team2 = ?)
               OR (team1 = ? AND team2 = ?)
        """, [team1, team2,
              team1, team2,
              team2, team1,
              team1, team2,
              team2, team1]).fetchone()

        if not row or row[0] == 0:
            return {"total_matches": 0, "team1_wins": 0, "team2_wins": 0,
                    "team1_win_pct": 0.0, "team2_win_pct": 0.0}

        total = row[0]
        return {
            "team1": team1,
            "team2": team2,
            "total_matches": total,
            "team1_wins": row[1],
            "team2_wins": row[2],
            "team1_win_pct": round(row[1] * 100.0 / total, 1),
            "team2_win_pct": round(row[2] * 100.0 / total, 1),
            "team1_avg_score": row[3],
            "team2_avg_score": row[4],
        }
    except Exception as e:
        logger.error("get_head_to_head failed: %s", e)
        return {"total_matches": 0, "team1_wins": 0, "team2_wins": 0,
                "team1_win_pct": 0.0, "team2_win_pct": 0.0}


def get_team_stats(team: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) AS wins,
                ROUND(SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END)
                      * 100.0 / COUNT(*), 1) AS win_rate,
                ROUND(AVG(CASE WHEN batting_first = ? THEN first_innings_score
                               ELSE second_innings_score END), 1) AS avg_score
            FROM matches
            WHERE team1 = ? OR team2 = ?
        """, [team, team, team, team, team]).fetchone()

        if not row or row[0] == 0:
            return {}
        return {
            "team": team,
            "total_matches": row[0],
            "wins": row[1],
            "win_rate": row[2],
            "avg_score": row[3],
        }
    except Exception as e:
        logger.error("get_team_stats failed: %s", e)
        return {}


def get_all_teams() -> list:
    conn = get_connection()
    rows = conn.execute("""
        SELECT DISTINCT team1 FROM matches
        UNION
        SELECT DISTINCT team2 FROM matches
        ORDER BY 1
    """).fetchall()
    return [r[0] for r in rows]


def get_all_venues() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT venue FROM matches ORDER BY venue"
    ).fetchall()
    return [r[0] for r in rows]


def compute_team_strengths() -> dict[str, float]:
    """Compute team win rates from actual data (used on startup)."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            WITH team_matches AS (
                SELECT team1 AS team FROM matches
                UNION ALL
                SELECT team2 FROM matches
            ),
            team_wins AS (
                SELECT winner AS team, COUNT(*) AS wins FROM matches GROUP BY winner
            ),
            team_played AS (
                SELECT team, COUNT(*) AS played FROM team_matches GROUP BY team
            )
            SELECT p.team, COALESCE(w.wins, 0) * 1.0 / p.played AS win_rate
            FROM team_played p
            LEFT JOIN team_wins w ON p.team = w.team
            ORDER BY win_rate DESC
        """).fetchall()
        return {r[0]: round(r[1], 4) for r in rows}
    except Exception as e:
        logger.warning("compute_team_strengths failed: %s", e)
        return {}


def get_match_features(team1: str, team2: str,
                       venue: str, toss_winner: str,
                       toss_decision: str) -> Optional[dict]:
    """Return pre-computed stats for prediction feature engineering."""
    conn = get_connection()
    try:
        t1 = conn.execute("""
            SELECT
                ROUND(SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END)
                      * 100.0 / COUNT(*), 3) AS win_rate
            FROM matches WHERE team1 = ? OR team2 = ?
        """, [team1, team1, team1]).fetchone()

        t2 = conn.execute("""
            SELECT
                ROUND(SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END)
                      * 100.0 / COUNT(*), 3) AS win_rate
            FROM matches WHERE team1 = ? OR team2 = ?
        """, [team2, team2, team2]).fetchone()

        h2h = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) AS t1_wins
            FROM matches
            WHERE (team1 = ? AND team2 = ?) OR (team1 = ? AND team2 = ?)
        """, [team1, team1, team2, team2, team1]).fetchone()

        vn = conn.execute("""
            SELECT
                ROUND(AVG(first_innings_score), 1) AS avg_score,
                ROUND(SUM(CASE WHEN result = 'batting_first' THEN 1 ELSE 0 END)
                      * 100.0 / COUNT(*), 3) AS bf_win_rate
            FROM matches WHERE LOWER(venue) LIKE LOWER(?)
        """, [f"%{venue}%"]).fetchone()

        t1_wr = (t1[0] or 50.0) / 100.0
        t2_wr = (t2[0] or 50.0) / 100.0
        h2h_total = h2h[0] if h2h else 0
        h2h_t1 = (h2h[1] / h2h_total) if h2h_total > 0 else 0.5
        avg_score = (vn[0] or 162.0) if vn else 162.0
        bf_wr = (vn[1] or 50.0) / 100.0 if vn else 0.5

        return {
            "team1_win_rate": t1_wr,
            "team2_win_rate": t2_wr,
            "h2h_team1_win_rate": h2h_t1,
            "toss_winner_is_team1": 1 if toss_winner == team1 else 0,
            "toss_decision_bat": 1 if toss_decision.lower() == "bat" else 0,
            "venue_bf_win_rate": bf_wr,
            "avg_score_at_venue": avg_score,
        }
    except Exception as e:
        logger.error("get_match_features failed: %s", e)
        return None
