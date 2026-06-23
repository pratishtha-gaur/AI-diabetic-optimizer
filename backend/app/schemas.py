# backend/app/schemas.py
#
# WHAT THIS FILE DOES
# -------------------
# Defines the exact shape of every request coming IN and every
# response going OUT of the API. Pydantic validates these
# automatically — if a client sends a string where a float is
# expected, FastAPI rejects it with a clear error message before
# it ever reaches your model code.
#
# WHAT IS PYDANTIC?
# -----------------
# Pydantic is a data validation library. You describe the shape
# of your data as a Python class, and Pydantic ensures any data
# matching that class is valid. FastAPI is built on Pydantic —
# it uses these classes to auto-validate incoming JSON, auto-
# generate API docs, and auto-serialize outgoing responses.
#
# ANALOGY: think of schemas like a form with required fields.
# If someone submits the form without filling in "glucose_history",
# the form is rejected immediately — the model code never runs.

from pydantic import BaseModel, Field
from typing import List


# ──────────────────────────────────────────────
# REQUEST SCHEMAS — what the client sends IN
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input for POST /predict — the LSTM glucose forecaster.

    The client sends the last 12 glucose readings (1 hour of
    5-min readings) and the current hour. The LSTM uses this
    window to predict the next reading.

    Fields:
        glucose_history : list of last 12 glucose values in mg/dL
                          must be exactly 12 values (seq_len from Phase 2)
        current_hour    : current hour as decimal (8.5 = 8:30am)
    """
    glucose_history: List[float] = Field(
        ...,                        # ... means required (no default)
        min_length=12,
        max_length=12,
        description="Last 12 glucose readings in mg/dL (5-min intervals = 1 hour)",
        examples=[[120.0, 122.5, 125.0, 128.3, 132.1, 138.4,
                   142.0, 140.5, 138.2, 135.7, 132.0, 129.4]]
    )
    current_hour: float = Field(
        ...,
        ge=0.0,   # ge = greater than or equal to 0
        lt=24.0,  # lt = less than 24
        description="Current hour as decimal (8.5 = 8:30am)",
        examples=[8.5]
    )


class RecommendRequest(BaseModel):
    """
    Input for POST /recommend — the RL lifestyle agent.

    The client sends the current glucose state (5 numbers matching
    the RL environment's observation space from Phase 3).

    Fields:
        glucose_mgdl        : current glucose in mg/dL
        glucose_trend       : rate of change in mg/dL per 5-min step
        current_hour        : current hour as decimal
        insulin_on_board    : recent insulin still active (0.0–1.0 scale)
    """
    glucose_mgdl: float = Field(
        ...,
        ge=40.0,    # physiological minimum
        le=400.0,   # physiological maximum
        description="Current glucose reading in mg/dL",
        examples=[145.0]
    )
    glucose_trend: float = Field(
        ...,
        ge=-30.0,
        le=30.0,
        description="Rate of change in mg/dL per 5-min step (positive = rising)",
        examples=[3.5]
    )
    current_hour: float = Field(
        ...,
        ge=0.0,
        lt=24.0,
        description="Current hour as decimal",
        examples=[13.5]
    )
    insulin_on_board: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of insulin still active (0=none, 1=max)",
        examples=[0.1]
    )


# ──────────────────────────────────────────────
# RESPONSE SCHEMAS — what the API sends OUT
# ──────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    Output of POST /predict.

    Fields:
        predicted_glucose   : next glucose value in mg/dL (de-normalized from model output)
        confidence_range    : ±range in mg/dL (based on model's historical MAE of 2.51)
        status              : clinical interpretation string
    """
    predicted_glucose: float = Field(
        description="Predicted glucose in mg/dL for next 5-minute reading",
        examples=[138.2]
    )
    confidence_range: float = Field(
        description="Expected error margin in mg/dL (±MAE from evaluation)",
        examples=[2.51]
    )
    status: str = Field(
        description="Clinical status: 'in_range', 'high', 'low', 'critical_high', 'critical_low'",
        examples=["in_range"]
    )


class RecommendResponse(BaseModel):
    """
    Output of POST /recommend.

    Fields:
        action_id           : integer action code (0-3) from RL agent
        action_name         : human-readable action string
        reasoning           : plain English explanation of why
        urgency             : 'routine', 'moderate', 'urgent'
    """
    action_id: int = Field(
        description="Action code: 0=nothing, 1=small insulin, 2=large insulin, 3=carbs",
        examples=[1]
    )
    action_name: str = Field(
        description="Human-readable action name",
        examples=["Take small correction insulin"]
    )
    reasoning: str = Field(
        description="Plain English explanation of the recommendation",
        examples=["Glucose is trending upward at 13.5pm — a small correction may help prevent a post-lunch spike."]
    )
    urgency: str = Field(
        description="How urgently to act: routine / moderate / urgent",
        examples=["moderate"]
    )


class HealthResponse(BaseModel):
    """Output of GET /health — confirms API and models are loaded."""
    status: str
    lstm_loaded: bool
    rl_agent_loaded: bool
    message: str