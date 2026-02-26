import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
from typing import Dict, Set, Union, List

from src.candidate_generation import (
    build_cooccurrence_matrix,
    compute_item_popularity,
    recommend_item_based,
)
from src.ranking.dataset import (
    fast_compute_features, 
    load_item_metadata, 
    build_user_category_profiles
)
from src.taxonomy import TaxonomyEngine


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERACTIONS_PATH   = os.path.join(BASE_DIR, "data", "processed", "interactions.csv")
LR_MODEL_PATH       = os.path.join(BASE_DIR, "artifacts", "models", "lr_ranker.joblib")
XGB_MODEL_PATH      = os.path.join(BASE_DIR, "artifacts", "models", "xgb_ranker.json")

# 6 Features matching your trained models
FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "item_interaction_count",       
    "user_history_count",           
    "user_category_affinity",       
    "time_since_last_interaction",  
]

N_CANDIDATES = 100     
K_FINAL      = 10      


# -------------------------------------------------------
# Shared State 
# -------------------------------------------------------
class RankingContext:
    """
    Holds every pre-computed artifact needed to score candidates.
    Contains similarity matrix, taxonomy engine, item stats, and user history.
    """

    def __init__(self, train_df: pd.DataFrame, min_cooccurrence: int = 2):
        print("Building RankingContext …")

        if not pd.api.types.is_datetime64_any_dtype(train_df["last_interaction_ts"]):
            train_df = train_df.copy()
            train_df["last_interaction_ts"] = pd.to_datetime(train_df["last_interaction_ts"])
            
        self.max_train_ts = train_df["last_interaction_ts"].max()

        # 1. Similarity
        self.similarity_matrix = build_cooccurrence_matrix(
            interactions=train_df, 
            min_cooccurrence=min_cooccurrence
        )
        print(f"  similarity_matrix: {len(self.similarity_matrix):,} items")

        # 2. Taxonomy & Metadata
        self.taxonomy_engine = TaxonomyEngine(train_df)
        self.item_categories = load_item_metadata()
        self.user_category_profiles = build_user_category_profiles(train_df, self.item_categories)

        # 3. Popularity & Stats
        pop_df = compute_item_popularity(train_df)
        max_score = pop_df["interaction_score"].max()
        max_count = pop_df["interaction_count"].max()
        
        self.item_stats_dict = {}
        for row in pop_df.itertuples():
            self.item_stats_dict[row.item_id] = {
                'score': row.interaction_score / max_score if max_score > 0 else 0,
                'count': row.interaction_count / max_count if max_count > 0 else 0
            }
        self.top_popular_items = pop_df["item_id"].tolist()

        # 4. User History
        self.user_histories = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.user_last_ts = train_df.groupby("user_id")["last_interaction_ts"].max().to_dict()
        print("  RankingContext ready.")


# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
def load_model(model_path: str):
    """
    Loads either a LogisticRegression (.joblib) or XGBoost (.json) model.
    Returns a wrapper that exposes a consistent .score(X) method.
    """
    class _ModelWrapper:
        def __init__(self, path: str):
            if path.endswith(".joblib"):
                self._model = joblib.load(path)
                # Monkey patch for sklearn versions if needed
                inner = self._model.named_steps['model'] if hasattr(self._model, 'named_steps') else self._model
                if not hasattr(inner, 'multi_class'): inner.multi_class = 'auto'
                self._kind = "lr"
            else:
                self._model = xgb.Booster()
                self._model.load_model(path)
                self._kind = "xgb"

        def score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self._kind == "lr":
                # Scikit-learn Pipeline handles DataFrames natively
                return self._model.predict_proba(X)[:, 1]
            
            # XGBoost handles DataFrames natively
            return self._model.predict(xgb.DMatrix(X, feature_names=FEATURE_COLUMNS))

    return _ModelWrapper(model_path)


# -------------------------------------------------------
# Core Inference
# -------------------------------------------------------
def infer(
    user_id: int, 
    ctx: RankingContext, 
    model, 
    n_candidates: int = N_CANDIDATES, 
    k: int = K_FINAL
) -> List[int]:
    """
    Full inference pipeline for a single user.
    1. Generate candidates (CF + Taxonomy Backfill).
    2. Compute 6 features for each candidate.
    3. Score using Ranker.
    4. Sort and return Top-K.
    """
    user_history = ctx.user_histories.get(user_id, set())
    user_last    = ctx.user_last_ts.get(user_id, pd.NaT)

    # 1. Candidate generation (CF + Taxonomy)
    candidates = recommend_item_based(
        user_history=user_history,
        similarity_matrix=ctx.similarity_matrix,
        top_popular_items=ctx.top_popular_items,
        k=n_candidates,
        taxonomy_engine=ctx.taxonomy_engine
    )

    if not candidates:
        return []

    # 2. Feature computation
    now = ctx.max_train_ts

    feature_rows = [
        fast_compute_features(
            row_user_id          = user_id,
            row_item_id          = item_id,
            row_ts               = now,
            row_weight           = 0.0,
            user_history_set     = user_history,
            user_last_ts         = user_last,
            similarity_matrix    = ctx.similarity_matrix,
            item_stats_dict      = ctx.item_stats_dict, 
            item_categories      = ctx.item_categories,        # Pass metadata
            user_category_profiles = ctx.user_category_profiles, # Pass profiles
            is_negative          = True, 
        )
        for item_id in candidates
    ]

    feature_df = pd.DataFrame(feature_rows)
    
    # Pass DataFrame directly to preserve column names for Model (e.g. StandardScaler)
    X = feature_df[FEATURE_COLUMNS]

    # 3. Score
    try:
        scores = model.score(X)
    except Exception as e:
        print(f"Error during scoring for user {user_id}: {e}")
        return candidates[:k]

    # 4. Sort
    feature_df["_score"] = scores
    top_k = feature_df.nlargest(k, "_score")["item_id"].tolist()
    return top_k


# -------------------------------------------------------
# Batch Helper
# -------------------------------------------------------
def infer_batch(user_ids: List[int], ctx: RankingContext, model, n_candidates: int = N_CANDIDATES, k: int = K_FINAL) -> Dict[int, List[int]]:
    """Helper to run inference on a batch of users."""
    results = {}
    for i, uid in enumerate(user_ids):
        results[uid] = infer(uid, ctx, model, n_candidates, k)
        if (i + 1) % 1000 == 0:
            print(f"  Scored {i + 1:,}/{len(user_ids):,} users")
    return results

if __name__ == "__main__":
    if os.path.exists(INTERACTIONS_PATH) and os.path.exists(LR_MODEL_PATH):
        interactions = pd.read_csv(INTERACTIONS_PATH)
        # Smoke test
        ctx = RankingContext(interactions.head(1000)) 
        model = load_model(LR_MODEL_PATH)
        print(f"Model loaded. Expecting {len(FEATURE_COLUMNS)} features.")
        sample_user = interactions["user_id"].iloc[0]
        recs = infer(sample_user, ctx, model)
        print(f"User {sample_user} recommendations: {recs}")