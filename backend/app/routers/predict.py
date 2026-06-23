# backend/app/routers/predict.py
#
# WHAT THIS FILE DOES
# -------------------
# Defines the POST /predict HTTP endpoint. This is the HTTP layer —
# it receives a validated request, calls the LSTM service, and
# returns a structured response. It deliberately contains zero
# ML logic — all of that lives in services/predictor.py.
#
# WHAT IS A ROUTER?
# -----------------
# FastAPI lets you split your endpoints across multiple files using
# APIRouter. Each router handles a group of related endpoints.
# main.py then mounts all routers together into one app.
# This is the FastAPI equivalent of "separation of concerns".

from fastapi import APIRouter, HTTPException
from ..schemas import PredictRequest, PredictResponse
from ..services.predictor import predictor

# Create a router — this is like a mini-app for /predict endpoints
router = APIRouter(
    prefix="/predict",    # all endpoints in this file start with /predict
    tags=["Prediction"],  # groups this endpoint in the auto-generated Swagger UI
)


@router.post(
    "",               # "" means the endpoint is exactly /predict (no suffix)
    response_model=PredictResponse,
    summary="Predict next glucose value",
    description=(
        "Given the last 12 glucose readings (1 hour of 5-minute intervals) "
        "and the current hour, returns the predicted glucose value for the "
        "next 5-minute reading using the trained LSTM model."
    ),
)
async def predict_glucose(request: PredictRequest) -> PredictResponse:
    """
    POST /predict

    FastAPI automatically:
      1. Parses the incoming JSON body
      2. Validates it against PredictRequest (via Pydantic)
      3. Calls this function with a typed Python object
      4. Serializes our return value back to JSON

    If validation fails (e.g. missing field, wrong type, glucose_history
    length != 12), FastAPI returns a 422 error with a clear message
    BEFORE this function is ever called.
    """
    try:
        result = predictor.predict(
            glucose_history=request.glucose_history,
            current_hour=request.current_hour,
        )
        return PredictResponse(**result)

    except RuntimeError as e:
        # Model not loaded — startup issue
        raise HTTPException(
            status_code=503,  # 503 = Service Unavailable
            detail=f"LSTM model not available: {str(e)}"
        )
    except Exception as e:
        # Unexpected error — return 500 with details
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )