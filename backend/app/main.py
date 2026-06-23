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

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

from .schemas import HealthResponse
from .services.predictor import predictor
from .services.rl_agent import recommender
from .routers import predict, recommend


# ──────────────────────────────────────────────
# Model paths — relative to project root
# ──────────────────────────────────────────────
LSTM_MODEL_PATH = os.path.join(root, "models", "saved", "glucose_lstm.pt")
RL_MODEL_PATH   = os.path.join(root, "models", "saved", "glucose_rl_agent.zip")


# ──────────────────────────────────────────────
# Lifespan — runs at startup and shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything BEFORE 'yield' runs at startup.
    Everything AFTER 'yield' runs at shutdown.

    This is where we load both models into memory.
    FastAPI guarantees this runs before any request is served.
    """
    print("🚀 Starting AI Diabetic Lifestyle Optimizer API...")

    #The lifespan startup pattern is still used to load models once into memory, 
    # but the exception handling changes the behavior 
    # from strict fast failure to graceful degradation.
    
    # Load LSTM predictor
    try:
        predictor.load(LSTM_MODEL_PATH)
    except FileNotFoundError:
        print(f"⚠️  LSTM model not found at {LSTM_MODEL_PATH}")
        print("   Run: python models/lstm/train.py")

    # Load RL agent
    try:
        recommender.load(RL_MODEL_PATH)
    except FileNotFoundError:
        print(f"⚠️  RL agent not found at {RL_MODEL_PATH}")
        print("   Run: python models/rl/train_agent.py")

    print("✅ API ready — visit http://localhost:8000/docs for Swagger UI\n")

    yield  # ← server runs here, handling requests

    # Shutdown cleanup (nothing needed for our simple models)
    print("👋 Shutting down...")


# ──────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────
app = FastAPI(
    title       = "AI Diabetic Lifestyle Optimizer",
    description = (
        "An AI-powered API that predicts short-term glucose levels (LSTM) "
        "and provides safe lifestyle recommendations (PPO Reinforcement Learning) "
        "for Type 1 diabetic users."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ──────────────────────────────────────────────
# CORS middleware
# ──────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) allows your React frontend
# (running on localhost:3000) to call this API (on localhost:8000).
# Without this, browsers block the request for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ──────────────────────────────────────────────
# Mount routers
# ──────────────────────────────────────────────
# Each router handles a group of related endpoints.
# The prefix here + prefix in the router = full path.
# e.g. app prefix="" + router prefix="/predict" → /predict
app.include_router(predict.router)
app.include_router(recommend.router)


# ──────────────────────────────────────────────
# Health check endpoint
# ──────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check API and model status",
)
async def health_check() -> HealthResponse:
    """
    GET /health

    Returns the status of the API and whether both ML models
    loaded successfully at startup. In production, monitoring
    systems call this endpoint to verify the service is alive.
    """
    lstm_ok = predictor.loaded
    rl_ok   = recommender.loaded

    if lstm_ok and rl_ok:
        status  = "healthy"
        message = "Both models loaded and ready"
    elif lstm_ok or rl_ok:
        status  = "degraded"
        message = "One model unavailable — partial functionality only"
    else:
        status  = "unhealthy"
        message = "No models loaded — run train.py scripts first"

    return HealthResponse(
        status           = status,
        lstm_loaded      = lstm_ok,
        rl_agent_loaded  = rl_ok,
        message          = message,
    )


# ──────────────────────────────────────────────
# Root endpoint
# ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "AI Diabetic Lifestyle Optimizer API",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
    }