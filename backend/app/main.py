# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="AI Diabetic Lifestyle Optimizer API",
    description="A test FastAPI setup for your AI backend",
    version="1.0.0"
)

# Request model using Pydantic
class UserData(BaseModel):
    name: str
    carbs: float
    activity_min: int

# Default route (GET)
@app.get("/")
def home():
    return {"message": "🎯 FastAPI + Pydantic v2 are running successfully!"}

# Test route (POST)
@app.post("/analyze")
def analyze(data: UserData):
    score = (data.activity_min * 0.5) - (data.carbs * 0.2)
    suggestion = "Good balance! 💪" if score > 0 else "Consider a light walk 🚶‍♀️"
    return {
        "user": data.name,
        "wellness_score": round(score, 2),
        "suggestion": suggestion
    }
