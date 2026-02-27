import sys
import joblib
import pandas as pd
from pathlib import Path

# --- Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.ranking.infer import RankingContext
from src.ranking.evaluation import time_based_split

INTERACTIONS_PATH = BASE_DIR / "data" / "processed" / "interactions.csv"
API_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "api"

def export_api_artifacts():
    print(f"1. Loading raw interactions data from {INTERACTIONS_PATH}...")
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(f"Cannot find data at {INTERACTIONS_PATH}")
        
    interactions = pd.read_csv(INTERACTIONS_PATH)
    
    # =====================================================================
    # MEMORY OPTIMIZATION FOR RENDER FREE TIER (512MB LIMIT)
    # =====================================================================
    print(f"  Original size: {len(interactions):,} rows.")
    
    # Sort by time to keep the most recent data
    interactions['last_interaction_ts'] = pd.to_datetime(interactions['last_interaction_ts'])
    interactions = interactions.sort_values('last_interaction_ts')
    
    # Keep only the last 250,000 interactions to strictly ensure it fits in 512MB RAM
    interactions = interactions.tail(250000).copy()
    print(f"  Truncated size: {len(interactions):,} rows.")
    # =====================================================================

    train_df, _ = time_based_split(interactions)
    
    print("2. Building RankingContext...")
    ctx = RankingContext(train_df)
    
    print(f"3. Exporting lightweight artifacts to {API_ARTIFACTS_DIR}...")
    API_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(ctx.similarity_matrix, API_ARTIFACTS_DIR / "similarity_matrix.joblib", compress=3)
    joblib.dump(ctx.taxonomy_engine, API_ARTIFACTS_DIR / "taxonomy_engine.joblib", compress=3)
    joblib.dump(ctx.item_categories, API_ARTIFACTS_DIR / "item_categories.joblib", compress=3)
    joblib.dump(ctx.user_category_profiles, API_ARTIFACTS_DIR / "user_category_profiles.joblib", compress=3)
    joblib.dump(ctx.item_stats_dict, API_ARTIFACTS_DIR / "item_stats_dict.joblib", compress=3)
    joblib.dump(ctx.top_popular_items, API_ARTIFACTS_DIR / "top_popular_items.joblib", compress=3)
    joblib.dump(ctx.user_histories, API_ARTIFACTS_DIR / "user_histories.joblib", compress=3)
    joblib.dump(ctx.user_last_ts, API_ARTIFACTS_DIR / "user_last_ts.joblib", compress=3)
    joblib.dump(ctx.max_train_ts, API_ARTIFACTS_DIR / "max_train_ts.joblib")
    
    print("\nExport Complete! Fast inference artifacts are ready.")
    
    # Print out some valid user IDs so you know who to test with!
    valid_users = list(ctx.user_histories.keys())[:5]
    print("\n" + "="*50)
    print(f"SUCCESS! When testing your API on Render, use one of these User IDs:")
    print(valid_users)
    print("="*50)

if __name__ == "__main__":
    export_api_artifacts()