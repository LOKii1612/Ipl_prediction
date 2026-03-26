"""Pydantic request / response models for the IPL Prediction API."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re


# ─── Request Models ────────────────────────────────────────────────────────────

class MatchPredictionRequest(BaseModel):
    team1: str = Field(..., description="Team 1 abbreviation", json_schema_extra={"example": "MI"})
    team2: str = Field(..., description="Team 2 abbreviation", json_schema_extra={"example": "CSK"})
    venue: str = Field(..., json_schema_extra={"example": "Wankhede Stadium, Mumbai"})
    toss_winner: str = Field(..., json_schema_extra={"example": "MI"})
    toss_decision: str = Field(..., description="'bat' or 'field'", json_schema_extra={"example": "bat"})

    @field_validator("team1", "team2", "toss_winner")
    @classmethod
    def validate_team_code(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.match(r"^[A-Z]{2,5}$", v):
            raise ValueError("Team code must be 2-5 uppercase letters")
        return v

    @field_validator("toss_decision")
    @classmethod
    def validate_toss_decision(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("bat", "field"):
            raise ValueError("toss_decision must be 'bat' or 'field'")
        return v

    @field_validator("venue")
    @classmethod
    def validate_venue(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3 or len(v) > 200:
            raise ValueError("Venue must be 3-200 characters")
        return v


class TossImpactRequest(BaseModel):
    venue: str = Field(..., json_schema_extra={"example": "Wankhede Stadium, Mumbai"})
    toss_decision: str = Field(..., description="'bat' or 'field'", json_schema_extra={"example": "field"})
    team1: Optional[str] = Field(None, json_schema_extra={"example": "MI"})
    team2: Optional[str] = Field(None, json_schema_extra={"example": "CSK"})

    @field_validator("toss_decision")
    @classmethod
    def validate_toss_decision(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("bat", "field"):
            raise ValueError("toss_decision must be 'bat' or 'field'")
        return v

    @field_validator("venue")
    @classmethod
    def validate_venue(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3 or len(v) > 200:
            raise ValueError("Venue must be 3-200 characters")
        return v


class PlayerPerformanceRequest(BaseModel):
    player_name: str = Field(..., json_schema_extra={"example": "Rohit Sharma"})
    team: str = Field(..., json_schema_extra={"example": "MI"})
    venue: str = Field(..., json_schema_extra={"example": "Wankhede Stadium, Mumbai"})
    role: str = Field(..., description="batsman / bowler / allrounder", json_schema_extra={"example": "batsman"})

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("batsman", "bowler", "allrounder"):
            raise ValueError("role must be 'batsman', 'bowler', or 'allrounder'")
        return v


# ─── Response Models ───────────────────────────────────────────────────────────

class WinnerPrediction(BaseModel):
    predicted_winner: str
    win_probability: float
    confidence: str
    team1: str
    team2: str
    team1_win_probability: float
    team2_win_probability: float
    key_factors: list[str]


class TossImpactResponse(BaseModel):
    venue: str
    toss_decision: str
    batting_first_win_probability: float
    chasing_win_probability: float
    recommended_decision: str
    toss_impact_score: float
    analysis: str


class VenueStatsResponse(BaseModel):
    venue: str
    total_matches: int
    avg_first_innings_score: float
    avg_second_innings_score: float
    batting_first_win_pct: float
    avg_margin: float
    recommendation: str


class HeadToHeadResponse(BaseModel):
    team1: str
    team2: str
    total_matches: int
    team1_wins: int
    team2_wins: int
    team1_win_pct: float
    team2_win_pct: float
    team1_avg_score: Optional[float]
    team2_avg_score: Optional[float]
    dominant_team: str


class PlayerPerformanceResponse(BaseModel):
    player_name: str
    team: str
    venue: str
    predicted_performance_score: float
    performance_tier: str
    impact_rating: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    database_connected: bool
    total_matches_in_db: int
    available_teams: list[str]
    version: str
