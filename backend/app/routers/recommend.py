# backend/app/routers/recommend.py
#
# WHAT THIS FILE DOES
# -------------------
# Defines the POST /recommend HTTP endpoint. Same structure as
# predict.py — pure HTTP layer, zero ML logic.

from fastapi import APIRouter, HTTPException
from ..schemas import RecommendRequest, RecommendResponse
from ..services.rl_agent import recommender

router = APIRouter(
    prefix="/recommend",
    tags=["Recommendation"],
)


@router.post(
    "",
    response_model=RecommendResponse,
    summary="Get lifestyle recommendation",
    description=(
        "Given the current glucose state (level, trend, time, insulin on board), "
        "returns a lifestyle recommendation from the trained PPO reinforcement "
        "learning agent — one of: do nothing, take small correction insulin, "
        "take large correction insulin, or eat fast-acting carbs."
    ),
)
async def recommend_action(request: RecommendRequest) -> RecommendResponse:
    """
    POST /recommend

    The RL agent sees the same 5-number state vector it was trained on
    in Phase 3, built from the request's fields in services/rl_agent.py.
    """
    try:
        result = recommender.recommend(
            glucose_mgdl     = request.glucose_mgdl,
            glucose_trend    = request.glucose_trend,
            current_hour     = request.current_hour,
            insulin_on_board = request.insulin_on_board,
        )
        return RecommendResponse(**result)

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"RL agent not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )