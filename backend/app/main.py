# backend/app/main.py
#
# WHAT THIS FILE DOES
# -------------------
# The entry point of the entire backend. It:
#   1. Creates the FastAPI app with metadata
#   2. Loads both ML models ONCE at startup (not per-request)
#   3. Mounts the predict and recommend routers
#   4. Provides a /health endpoint to verify everything is loaded
#
# THE LIFESPAN PATTERN
# --------------------
# FastAPI uses a "lifespan" context manager to run code at startup
# and shutdown. Model loading goes here because:
#   - It happens ONCE before any requests are served
#   - If loading fails, the server fails to start (fast failure)
#   - The loaded model stays in memory for the server's lifetime
#
# This is the standard production pattern for ML APIs — never
# load models inside request handlers.

# backend/app/main.py  (production-ready version)
#
# Changes from Phase 4:
#   1. CORS origins read from environment variable
#   2. Serves React frontend's built static files
#      so one server handles both API and UI

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

from .schemas import HealthResponse
from .services.predictor import predictor
from .services.rl_agent import recommender
from .routers import predict, recommend

LSTM_MODEL_PATH = os.path.join(root, "models", "saved", "glucose_lstm.pt")
RL_MODEL_PATH   = os.path.join(root, "models", "saved", "glucose_rl_agent.zip")

# ── Frontend static files path ─────────────────────────────────────
FRONTEND_DIST = os.path.join(root, "frontend", "dist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting AI Diabetic Lifestyle Optimizer API...")
    try:
        predictor.load(LSTM_MODEL_PATH)
    except FileNotFoundError:
        print(f"⚠️  LSTM model not found. Run: python train_all.py")

    try:
        recommender.load(RL_MODEL_PATH)
    except FileNotFoundError:
        print(f"⚠️  RL agent not found. Run: python train_all.py")

    print("✅ API ready\n")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title       = "AI Diabetic Lifestyle Optimizer",
    description = "LSTM glucose forecasting + PPO lifestyle recommendations",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS — reads from environment variable in production ────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── API routers ─────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(recommend.router)


# ── Health endpoint ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    lstm_ok = predictor.loaded
    rl_ok   = recommender.loaded
    if lstm_ok and rl_ok:
        status, message = "healthy", "Both models loaded and ready"
    elif lstm_ok or rl_ok:
        status, message = "degraded", "One model unavailable"
    else:
        status, message = "unhealthy", "No models loaded"
    return HealthResponse(status=status, lstm_loaded=lstm_ok, rl_agent_loaded=rl_ok, message=message)


# ── Serve React frontend (production) ──────────────────────────────
# In production (inside Docker), the built React app lives at
# frontend/dist/. We mount it so the same server handles both
# the API (/predict, /recommend) and the web UI (/).
if os.path.exists(FRONTEND_DIST):
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    # Catch-all: serve index.html for any non-API route
    # This makes React Router work correctly
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        index = os.path.join(FRONTEND_DIST, "index.html")
        return FileResponse(index)
else:
    @app.get("/", tags=["Health"])
    async def root():
        return {"message": "AI Diabetic Lifestyle Optimizer API", "docs": "/docs"}