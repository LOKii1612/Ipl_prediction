# IPL Cricket Prediction Bot

A FastAPI service that predicts IPL T20 match outcomes using five XGBoost models trained on synthetic historical data stored in DuckDB.

## Features

- **Match Winner Prediction** — predicts which team wins given toss and venue context
- **Toss Impact Analysis** — quantifies how toss decision affects win probability at a specific venue
- **Player Performance Scoring** — predicts a composite performance score for any player/role
- **Venue Statistics** — historical batting-first win rates, average scores, and recommendations
- **Head-to-Head Records** — win rates, average scores, and dominant team for any pair of IPL franchises
- **5 XGBoost Models** — match_winner, toss_impact, batting_first, player_performance, venue_advantage
- **DuckDB Backend** — fast in-process analytical queries with no external database server
- **Graceful Fallback** — all prediction endpoints work even before models are trained (heuristic fallback)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + Pydantic v2 |
| ML models | XGBoost 2.x (classifier + regressor) |
| Database | DuckDB (embedded, file-based) |
| Server | Uvicorn (ASGI) |
| Data wrangling | Pandas + NumPy |
| Testing | Pytest |
| Containerisation | Docker (multi-stage build) |
| Deployment | Railway (`railway up`) |

## Project Structure

```
ipl-bot/
├── src/
│   ├── main.py          # FastAPI app & route handlers
│   ├── predictor.py     # IPLPredictor class (model loading + inference)
│   ├── database.py      # DuckDB query helpers
│   └── models.py        # Pydantic request/response schemas
├── scripts/
│   ├── generate_synthetic_data.py   # Creates ipl_matches.duckdb (1 200 matches)
│   └── train_models.py              # Trains & saves 5 XGBoost models
├── models/              # Saved model JSON files (git-ignored)
├── data/                # DuckDB file + training report (git-ignored)
├── tests/
│   ├── test_api.py      # FastAPI endpoint tests
│   └── test_models.py   # XGBoost model validation tests
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate synthetic data

```bash
python3 scripts/generate_synthetic_data.py
# Creates data/ipl_matches.duckdb with 1 200 matches, 70 players
```

### Train models

```bash
python3 scripts/train_models.py
# Saves 5 models to models/ and a training_report.json to data/
```

### Run the API locally

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive docs: <http://localhost:8000/docs>

## API Endpoints

### POST /predict/match

Predict the winner of an IPL match.

```bash
curl -X POST http://localhost:8000/predict/match \
  -H "Content-Type: application/json" \
  -d '{
    "team1": "MI",
    "team2": "CSK",
    "venue": "Wankhede Stadium, Mumbai",
    "toss_winner": "MI",
    "toss_decision": "bat"
  }'
```

**Response:**
```json
{
  "predicted_winner": "MI",
  "win_probability": 0.582,
  "confidence": "Medium",
  "team1_win_probability": 0.582,
  "team2_win_probability": 0.418,
  "key_factors": ["MI has stronger overall record"]
}
```

### POST /predict/toss-impact

Analyse how a toss decision affects win probability at a venue.

```bash
curl -X POST http://localhost:8000/predict/toss-impact \
  -H "Content-Type: application/json" \
  -d '{
    "venue": "MA Chidambaram Stadium, Chennai",
    "toss_decision": "field",
    "team1": "CSK",
    "team2": "MI"
  }'
```

**Response:**
```json
{
  "venue": "MA Chidambaram Stadium, Chennai",
  "toss_decision": "field",
  "batting_first_win_probability": 0.44,
  "chasing_win_probability": 0.56,
  "recommended_decision": "field",
  "toss_impact_score": 0.12,
  "analysis": "Mild venue trend — batting first wins 44% here."
}
```

### POST /predict/player-performance

Predict a player's composite performance score.

```bash
curl -X POST http://localhost:8000/predict/player-performance \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Jasprit Bumrah",
    "team": "MI",
    "venue": "Wankhede Stadium, Mumbai",
    "role": "bowler"
  }'
```

**Response:**
```json
{
  "player_name": "Jasprit Bumrah",
  "team": "MI",
  "venue": "Wankhede Stadium, Mumbai",
  "predicted_performance_score": 54.2,
  "performance_tier": "Strong",
  "impact_rating": "High impact"
}
```

### GET /stats/venue/{venue}

Historical stats for a venue (partial name, case-insensitive).

```bash
curl http://localhost:8000/stats/venue/Wankhede
```

**Response:**
```json
{
  "venue": "Wankhede Stadium, Mumbai",
  "total_matches": 87,
  "avg_first_innings_score": 168.3,
  "avg_second_innings_score": 155.1,
  "batting_first_win_pct": 47.1,
  "avg_margin": 21.4,
  "recommendation": "Chase (batting-first wins 47.1% here)"
}
```

### GET /stats/head-to-head/{team1}/{team2}

Head-to-head record between two teams.

```bash
curl http://localhost:8000/stats/head-to-head/MI/CSK
```

**Response:**
```json
{
  "team1": "MI",
  "team2": "CSK",
  "total_matches": 34,
  "team1_wins": 18,
  "team2_wins": 16,
  "team1_win_pct": 52.9,
  "team2_win_pct": 47.1,
  "team1_avg_score": 172.4,
  "team2_avg_score": 168.1,
  "dominant_team": "MI"
}
```

### GET /stats/teams

List all IPL teams in the database.

```bash
curl http://localhost:8000/stats/teams
```

### GET /stats/venues

List all venues in the database.

```bash
curl http://localhost:8000/stats/venues
```

### GET /health

Health check — reports model and database status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "database_connected": true,
  "total_matches_in_db": 1200,
  "available_teams": ["CSK", "DC", "GT", "KKR", ...],
  "version": "1.0.0"
}
```

## Running Tests

```bash
# Run all tests (stats tests skip automatically if DB is absent)
pytest tests/ -v

# Run only API tests
pytest tests/test_api.py -v

# Run only model validation tests
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Docker

### Build the image

```bash
docker build -t ipl-bot .
```

> **Note:** The `models/` and `data/` directories must exist and be populated before building.
> Run `generate_synthetic_data.py` and `train_models.py` first.

### Run the container

```bash
docker run -p 8000:8000 ipl-bot
```

### Environment variables at runtime

```bash
docker run -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  -e WORKERS=4 \
  ipl-bot
```

## Railway Deployment

1. **Install the Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login**:
   ```bash
   railway login
   ```

3. **Initialise the project** (first time only):
   ```bash
   railway init
   ```

4. **Generate data and train models locally**, then commit the artefacts:
   ```bash
   python3 scripts/generate_synthetic_data.py
   python3 scripts/train_models.py
   git add data/ipl_matches.duckdb models/*.json data/training_report.json
   git commit -m "Add trained models and database"
   ```

5. **Deploy**:
   ```bash
   railway up
   ```

   Railway automatically detects the `Dockerfile` and builds the container.

6. **Check the deployment**:
   ```bash
   railway status
   railway logs
   ```

## Environment Variables Reference

| Variable    | Default       | Description |
|-------------|---------------|-------------|
| `MODEL_DIR` | `./models`    | Path to directory containing model JSON files |
| `DATA_DIR`  | `./data`      | Path to directory containing the DuckDB file |
| `HOST`      | `0.0.0.0`     | Host address for uvicorn |
| `PORT`      | `8000`        | Port for uvicorn |
| `WORKERS`   | `1`           | Number of uvicorn worker processes |
| `LOG_LEVEL` | `INFO`        | Python logging level |
| `APP_ENV`   | `development` | Set to `production` to disable /docs |

Copy `.env.example` to `.env` and fill in values for local development.

## IPL Team Abbreviations

| Code | Team |
|------|------|
| MI   | Mumbai Indians |
| CSK  | Chennai Super Kings |
| GT   | Gujarat Titans |
| KKR  | Kolkata Knight Riders |
| SRH  | Sunrisers Hyderabad |
| RR   | Rajasthan Royals |
| DC   | Delhi Capitals |
| RCB  | Royal Challengers Bangalore |
| LSG  | Lucknow Super Giants |
| PBKS | Punjab Kings |
