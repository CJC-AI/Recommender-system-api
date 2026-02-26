import time
import logging
import pandas as pd
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from pathlib import Path

# Import your existing logic
# We assume the app is run from the root, so src. is accessible
from src.ranking.infer import RankingContext, load_model, infer

# --- Configuration ---
# distinct paths for interactions and model artifacts
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # move one level up from /app

INTERACTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "lr_ranker.joblib"

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recsys_api")

# --- Global State ---
# We store heavy objects here so they are loaded only once
ml_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts and data context on startup.
    """
    logger.info("Loading RankingContext and Models...")
    
    # 1. Load Data
    if not INTERACTIONS_PATH.exists():
        logger.error(f"Interactions file not found at {INTERACTIONS_PATH}")
        raise FileNotFoundError("Critical data missing.")
        
    df = pd.read_csv(INTERACTIONS_PATH)
    
    # 2. Build Context (Sim Matrix, Taxonomy, Popularity)
    # This might take a few seconds/minutes depending on data size
    ctx = RankingContext(df)
    
    # 3. Load Model
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError("Critical model missing.")
        
    model = load_model(str(MODEL_PATH))
    
    # Store in global state
    ml_context["ctx"] = ctx
    ml_context["model"] = model
    
    logger.info("System Ready!")
    yield
    # (Cleanup code goes here if needed)
    ml_context.clear()

# Initialize App
app = FastAPI(title="Recommender System API", lifespan=lifespan)

# --- Pydantic Models (Input/Output Validation) ---
class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    latency_ms: float

# --- Middleware: Latency Logging ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Log latency for every request
    logger.info(f"Path: {request.url.path} | Latency: {process_time:.2f}ms")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Simple check to see if API is alive"""
    if not ml_context:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "model_loaded": True}

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(payload: RecommendRequest):
    """
    Generate recommendations for a specific user.
    """
    start = time.time()
    
    try:
        # Retrieve loaded artifacts
        ctx = ml_context["ctx"]
        model = ml_context["model"]
        
        # Run Inference
        recs = infer(
            user_id=payload.user_id,
            ctx=ctx,
            model=model,
            k=payload.k
        )
        
        latency = (time.time() - start) * 1000
        return RecommendResponse(
            user_id=payload.user_id,
            recommendations=recs,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        logger.error(f"Error generating recs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run locally for debugging
    uvicorn.run(app, host="0.0.0.0", port=8000)