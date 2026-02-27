import sys
import joblib
import pandas as pd
from pathlib import Path

# --- Path Resolution with Pathlib ---
# __file__ is app/export_artifacts.py
# .parent is app/
# .parent.parent is the root directory (RECOMMENDER-SYSTEM-API)
BASE_DIR = Path(__file__).resolve().parent.parent

# Add root directory to sys.path so Python can find the 'src' module
sys.path.append(str(BASE_DIR))

from src.ranking.infer import RankingContext
from src.ranking.evaluation import time_based_split

# Define paths
INTERACTIONS_PATH = BASE_DIR / "data" / "processed" / "interactions.csv"
API_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "api"

def export_api_artifacts():
    print(f"1. Loading raw interactions data from {INTERACTIONS_PATH}...")
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(f"Cannot find data at {INTERACTIONS_PATH}")
        
    interactions = pd.read_csv(INTERACTIONS_PATH)
    train_df, _ = time_based_split(interactions)
    
    print("2. Building RankingContext (This takes heavy CPU & RAM)...")
    ctx = RankingContext(train_df)
    
    print(f"3. Exporting lightweight artifacts to {API_ARTIFACTS_DIR}...")
    # Create the directory if it doesn't exist (equivalent to os.makedirs(..., exist_ok=True))
    API_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dump dictionaries with compression
    joblib.dump(ctx.similarity_matrix, API_ARTIFACTS_DIR / "similarity_matrix.joblib", compress=3)
    joblib.dump(ctx.taxonomy_engine, API_ARTIFACTS_DIR / "taxonomy_engine.joblib", compress=3)
    joblib.dump(ctx.item_categories, API_ARTIFACTS_DIR / "item_categories.joblib", compress=3)
    joblib.dump(ctx.user_category_profiles, API_ARTIFACTS_DIR / "user_category_profiles.joblib", compress=3)
    joblib.dump(ctx.item_stats_dict, API_ARTIFACTS_DIR / "item_stats_dict.joblib", compress=3)
    joblib.dump(ctx.top_popular_items, API_ARTIFACTS_DIR / "top_popular_items.joblib", compress=3)
    joblib.dump(ctx.user_histories, API_ARTIFACTS_DIR / "user_histories.joblib", compress=3)
    joblib.dump(ctx.user_last_ts, API_ARTIFACTS_DIR / "user_last_ts.joblib", compress=3)
    joblib.dump(ctx.max_train_ts, API_ARTIFACTS_DIR / "max_train_ts.joblib")
    
    print("Export Complete! Fast inference artifacts are ready.")

if __name__ == "__main__":
    export_api_artifacts()