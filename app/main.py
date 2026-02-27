import sys
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from pathlib import Path

# --- Path Resolution with Pathlib ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.ranking.infer import load_model, infer

# Define paths
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "lr_ranker.joblib"
API_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "api"

# --- Pydantic Schemas ---
class RecommendationRequest(BaseModel):
    user_id: int
    n_candidates: int = 100
    top_k: int = 10

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[int]

# --- Lightweight Context Class ---
class LoadedRankingContext:
    """A memory-efficient class that mimics RankingContext using pre-computed dicts."""
    def __init__(self, artifacts_dir: Path):
        print("Loading pre-computed dictionaries into memory...")
        # joblib natively supports pathlib.Path objects
        self.similarity_matrix      = joblib.load(artifacts_dir / "similarity_matrix.joblib")
        self.taxonomy_engine        = joblib.load(artifacts_dir / "taxonomy_engine.joblib")
        self.item_categories        = joblib.load(artifacts_dir / "item_categories.joblib")
        self.user_category_profiles = joblib.load(artifacts_dir / "user_category_profiles.joblib")
        self.item_stats_dict        = joblib.load(artifacts_dir / "item_stats_dict.joblib")
        self.top_popular_items      = joblib.load(artifacts_dir / "top_popular_items.joblib")
        self.user_histories         = joblib.load(artifacts_dir / "user_histories.joblib")
        self.user_last_ts           = joblib.load(artifacts_dir / "user_last_ts.joblib")
        self.max_train_ts           = joblib.load(artifacts_dir / "max_train_ts.joblib")
        print("Dictionaries loaded successfully.")

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Lightweight API...")
    
    # 1. Load Pre-computed Data structures (Fast & Low RAM)
    if not API_ARTIFACTS_DIR.exists():
        raise FileNotFoundError(f"API artifacts missing. Run app/export_artifacts.py first! Looked in: {API_ARTIFACTS_DIR}")
        
    app.state.ctx = LoadedRankingContext(API_ARTIFACTS_DIR)
    
    # 2. Load Model
    # Explicitly casting to string because some underlying XGBoost/Sklearn 
    # load utilities in older versions prefer strings over Path objects.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    app.state.model = load_model(str(MODEL_PATH))
    
    print("Engine Ready. API is accepting requests.")
    yield

# --- App Definition ---
app = FastAPI(title="RetailRocket Recommender API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest, req: Request):
    try:
        recs = infer(
            user_id=request.user_id,
            ctx=req.app.state.ctx,
            model=req.app.state.model,
            n_candidates=request.n_candidates,
            k=request.top_k
        )
        return {"user_id": request.user_id, "recommendations": recs}
    except Exception as e:
        print(f"Error processing user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))