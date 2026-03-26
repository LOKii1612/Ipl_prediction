# 🏏 IPL Cricket Prediction Bot

AI-powered IPL match prediction system using XGBoost models, DuckDB, and FastAPI.

## Features

- **5 XGBoost Models:** Match winner, toss impact, batting first advantage, player performance, venue advantage
- **DuckDB Backend:** Fast analytical queries on 1,200+ match records
- **FastAPI REST API:** Clean endpoints with Pydantic validation
- **14 IPL Teams, 20+ Venues:** Full team and venue coverage
- **Dockerized:** Ready for Railway/Render/Fly deployment

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/rohith1729-inventor/ipl-bot.git
cd ipl-bot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate data & train models
python scripts/generate_synthetic_data.py
python scripts/train_models.py

# 3. Run the API
cp .env.example .env
uvicorn src.main:app --reload --port 8000
```

## API Endpoints

### Predict Match Winner
```bash
curl -X POST http://localhost:8000/predict/match \
  -H "Content-Type: application/json" \
  -d '{"team1": "MI", "team2": "CSK", "venue": "Wankhede Stadium, Mumbai", "toss_winner": "MI", "toss_decision": "bat"}'
```

### Toss Impact Analysis
```bash
curl -X POST http://localhost:8000/predict/toss-impact \
  -H "Content-Type: application/json" \
  -d '{"venue": "Wankhede Stadium, Mumbai", "toss_decision": "field", "team1": "MI", "team2": "CSK"}'
```

### Player Performance
```bash
curl -X POST http://localhost:8000/predict/player-performance \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Rohit Sharma", "team": "MI", "venue": "Wankhede Stadium, Mumbai", "role": "batsman"}'
```

### Venue Stats
```bash
curl http://localhost:8000/stats/venue/Wankhede
```

### Head-to-Head
```bash
curl http://localhost:8000/stats/head-to-head/MI/CSK
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t ipl-bot .
docker run -p 8000:8000 ipl-bot
```

## Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | XGBoost |
| Database | DuckDB |
| API | FastAPI + Uvicorn |
| Validation | Pydantic |
| Testing | pytest (109 tests) |
| Container | Docker (Python 3.11 slim) |

## Model Accuracy

| Model | Accuracy |
|-------|----------|
| Match Winner | 57.9% |
| Toss Impact | 50.8% |
| Batting First | 64.2% |
| Player Performance | R²=0.12 |
| Venue Advantage | R²=0.72 |

## License

MIT
