import pandas as pd
import numpy as np
import os
from typing import Dict, Set, List
from src.candidate_generation import build_cooccurrence_matrix, compute_item_popularity
from src.ranking.evaluation import time_based_split

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTERACTION_WEIGHTS = {
    1.0: 1.0,   # view
    3.0: 3.0,   # addtocart
    5.0: 5.0,   # transaction
}

def get_interaction_type_weight(interaction_score: float) -> float:
    return INTERACTION_WEIGHTS.get(interaction_score, interaction_score)

# -------------------------------------------------------
# Metadata & Profiling Helpers
# -------------------------------------------------------
def load_item_metadata() -> Dict[int, int]:
    """
    Loads item category mapping from BOTH property files.
    Returns: {item_id: category_id}
    """
    mapping = {}
    # Load both parts to ensure maximum coverage
    files = ["item_properties_part1.csv", "item_properties_part2.csv"]
    
    print("  Loading item metadata (Categories)...")
    for fname in files:
        path = os.path.join(BASE_DIR, "data", "raw", fname)
        if not os.path.exists(path):
            print(f"    [WARNING] {fname} not found. Skipping.")
            continue
            
        try:
            # Chunking not strictly necessary if memory allows, but good practice
            df = pd.read_csv(path)
            # Filter for categoryid property
            cat_df = df[df['property'] == 'categoryid'].copy()
            cat_df['value'] = pd.to_numeric(cat_df['value'], errors='coerce')
            cat_df = cat_df.dropna(subset=['value'])
            
            # Update mapping
            part_map = dict(zip(cat_df['itemid'], cat_df['value'].astype(int)))
            mapping.update(part_map)
            print(f"    Loaded {len(part_map)} mappings from {fname}.")
        except Exception as e:
            print(f"    [ERROR] Could not process {fname}: {e}")
            
    print(f"  Total item-category mappings: {len(mapping)}")
    return mapping

def build_user_category_profiles(interactions: pd.DataFrame, item_categories: Dict[int, int]) -> Dict[int, Dict[int, float]]:
    """
    Builds a normalized preference profile for each user.
    Example: {user_1: {cat_A: 0.8, cat_B: 0.2}}
    """
    print("  Building User Category Profiles...")
    
    user_cats = []
    for row in interactions.itertuples(index=False):
        cat = item_categories.get(row.item_id)
        if cat is not None:
            user_cats.append((row.user_id, cat))
            
    if not user_cats:
        return {}

    # Count category interactions per user
    df = pd.DataFrame(user_cats, columns=['user_id', 'category_id'])
    user_counts = df.groupby(['user_id', 'category_id']).size().reset_index(name='count')
    
    # Normalize by total user interactions
    user_totals = user_counts.groupby('user_id')['count'].transform('sum')
    user_counts['weight'] = user_counts['count'] / user_totals
    
    # Convert to fast lookup dict
    profiles = {}
    for row in user_counts.itertuples(index=False):
        if row.user_id not in profiles:
            profiles[row.user_id] = {}
        profiles[row.user_id][row.category_id] = row.weight
        
    return profiles

# -------------------------------------------------------
# Fast Feature Calculation
# -------------------------------------------------------
def fast_compute_features(
    row_user_id: int,
    row_item_id: int,
    row_ts: pd.Timestamp,
    row_weight: float,
    user_history_set: Set[int],
    user_last_ts: pd.Timestamp,
    similarity_matrix: Dict[int, Dict[int, float]],
    item_stats_dict: Dict[int, dict], 
    item_categories: Dict[int, int] = None,              
    user_category_profiles: Dict[int, Dict[int, float]] = None, 
    is_negative: bool = False
) -> dict:
    """
    Computes pointwise features for a single (user, item) pair.
    
    Features generated:
    1. item_similarity_score: Max similarity to user's history.
    2. item_popularity: Global popularity score (normalized).
    3. item_interaction_count: Global interaction count (normalized).
    4. user_history_count: Number of items in user's history.
    5. user_category_affinity: User's preference for item's category.
    6. time_since_last_interaction: Hours since user's last action.
    """
    
    # 1. Similarity Score (Optimized Intersection)
    max_sim = 0.0
    if user_history_set:
        neighbors = similarity_matrix.get(row_item_id, {})
        if neighbors:
            # Fast set intersection to find overlap between history and item's neighbors
            common_items = user_history_set.intersection(neighbors.keys())
            if common_items:
                max_sim = max(neighbors[i] for i in common_items)

    # 2. Time Since Last Interaction
    time_diff = 0.0
    if not pd.isnull(user_last_ts):
        time_diff = (row_ts - user_last_ts).total_seconds() / 3600.0
    
    # 3. Retrieve Pre-computed Item Stats
    stats = item_stats_dict.get(row_item_id, {'score': 0.0, 'count': 0.0})

    # 4. Category Affinity
    cat_affinity = 0.0
    if item_categories and user_category_profiles:
        item_cat = item_categories.get(row_item_id)
        if item_cat is not None:
            user_profile = user_category_profiles.get(row_user_id, {})
            cat_affinity = user_profile.get(item_cat, 0.0)

    return {
        "user_id": row_user_id,
        "item_id": row_item_id,
        "item_similarity_score": max_sim,
        "item_popularity": stats['score'],
        "item_interaction_count": stats['count'],      # FEATURE 3
        "user_history_count": len(user_history_set),   # FEATURE 4
        "user_category_affinity": cat_affinity,        # FEATURE 5
        "time_since_last_interaction": max(0.0, time_diff),
        "interaction_type_weight": 0.0 if is_negative else get_interaction_type_weight(row_weight),
        "label": 0 if is_negative else 1
    }

# -------------------------------------------------------
# Build Training Data
# -------------------------------------------------------
def build_training_data(
    train_df: pd.DataFrame,
    sample_users_frac: float = 0.1, 
    sample_items_top_n: int = 5000,    
    n_negatives: int = 6,              
    min_cooccurrence: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build training data with advanced negative sampling including Category Negatives.
    
    Negative Strategy:
    1. Hard Negatives: Items similar to user's history (Candidate Gen output).
    2. Category Negatives: Items from the SAME category as the positive sample.
    3. Popular Negatives: High-traffic items.
    4. Exclusion: Items user interacted with in Past OR Future.
    """
    
    print(f"Building TRAINING data with Category Negatives...")
    rng = np.random.RandomState(seed)
    
    # --- 1. Pre-processing ---
    train_df = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(train_df["last_interaction_ts"]):
        print("  Converting timestamps...")
        train_df["last_interaction_ts"] = pd.to_datetime(train_df["last_interaction_ts"])

    # User Sampling
    unique_users = train_df["user_id"].unique()
    sampled_users = rng.choice(
        unique_users, 
        size=int(len(unique_users) * sample_users_frac), 
        replace=False
    )
    train_df = train_df[train_df["user_id"].isin(set(sampled_users))].sort_values("last_interaction_ts")
    print(f"  Sampled {len(sampled_users)} users.")

    # --- 2. Build Artifacts ---
    print("  Building Matrix and Global History...")
    
    # Similarity Matrix (for Hard Negatives)
    similarity_matrix = build_cooccurrence_matrix(
        interactions=train_df, 
        min_cooccurrence=min_cooccurrence
    )
    
    # Popularity & Interaction Counts (for Features)
    pop_df = compute_item_popularity(train_df)
    
    # Compute maximums for normalization (0.0 - 1.0 range)
    # Ensure count feature play nicely with linear models if unscaled,
    # and keeps feature magnitudes consistent.
    max_score = pop_df["interaction_score"].max() if not pop_df.empty else 1.0
    max_count = pop_df["interaction_count"].max() if not pop_df.empty else 1.0

    # Build fast lookup dictionary for item stats
    item_stats_dict = {}
    for row in pop_df.itertuples():
        item_stats_dict[row.item_id] = {
            'score': row.interaction_score / max_score,
            'count': row.interaction_count / max_count
        }
    
    # Top N universe for popular sampling
    #popular_items_universe = pop_df.head(sample_items_top_n)["item_id"].values

    # All items univers for popular sampling (backup if top N is too restrictive)
    all_items_universe = pop_df["item_id"].values 

    # GLOBAL History (Past + Future) for strict exclusions
    # We must not sample an item as negative if the user will interact with it later
    user_global_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    # Load Metadata & Build Profiles
    item_categories = load_item_metadata()
    user_category_profiles = build_user_category_profiles(train_df, item_categories)

    # Pre-compute category -> items map for fast sampling (NEW)
    print("  Indexing items by category for negative sampling...")
    category_items_map = {}
    # Build a list of items for each category
    temp_cat_map = {} # cat_id -> list of item_ids
    for i_id, c_id in item_categories.items():
        if c_id not in temp_cat_map:
            temp_cat_map[c_id] = []
        temp_cat_map[c_id].append(i_id)
        
    # Convert to numpy arrays for fast choice
    for c_id in temp_cat_map:
        temp_cat_map[c_id] = np.array(temp_cat_map[c_id])
    category_items_map = temp_cat_map

    # --- 3. Row Generation ---
    print(f"  Generating rows (Target: {n_negatives} negatives per positive)...")
    training_data = []
    
    user_running_history = {} 
    user_last_ts = {}         
    
    for row in train_df.itertuples(index=False):
        u_id = row.user_id
        i_id = row.item_id
        score = row.interaction_score
        ts = row.last_interaction_ts
        
        current_history = user_running_history.get(u_id, set())
        last_ts = user_last_ts.get(u_id, pd.NaT)
        full_history_exclusion = user_global_history.get(u_id, set())
        
        # A. Positive Sample
        pos_row = fast_compute_features(
            u_id, i_id, ts, score, 
            current_history, last_ts, 
            similarity_matrix, item_stats_dict, 
            item_categories, user_category_profiles, 
            is_negative=False
        )
        training_data.append(pos_row)
        
        # B. Negative Samples
        selected_negatives = []
        
        # --- STRATEGY: 2 Hard, 2 Category, 2 Popular ---
        
        # 1. Hard Negatives (Similarity-based)
        # Look at items similar to what the user has recently seen
        hard_candidates = set()
        if current_history:
            # Sample up to 3 recent items to find neighbors for 
            # (Converting set to list is 0(N), but history is usually small)
            seed_items = rng.choice(list(current_history), size=min(len(current_history), 3), replace=False)
            for seed in seed_items:
                neighbors = similarity_matrix.get(seed, {})
                # Add top neighbor as hard negative candidates
                if neighbors:
                    # Sort by similarity and take 5 per seed
                    sorted_neighbors = sorted(neighbors, key=neighbors.get, reverse=True)[:5]
                    hard_candidates.update(sorted_neighbors)
        
        # Try to fill half the quota with Hard Negatives
        hard_quota = 2
        hard_candidates_list = list(hard_candidates)
        rng.shuffle(hard_candidates_list)

        for cand in hard_candidates_list:
            if len(selected_negatives) >= hard_quota: break

            # Strict Exclusion: Not in Past, Current, or Future
            if cand not in full_history_exclusion and cand != i_id:
                selected_negatives.append(cand)

        # 2. Category Negatives (Affinity-based) -> NEW
        # Sample items from the SAME category as the positive item (i_id)
        cat_quota = 4 # Target total 4 (2 Hard + 2 Cat)
        this_item_cat = item_categories.get(i_id)
        
        if this_item_cat is not None and this_item_cat in category_items_map:
            # Get all items in this category
            cat_items = category_items_map[this_item_cat]
            # Sample a few
            if len(cat_items) > 1:
                cat_negs = rng.choice(cat_items, size=min(len(cat_items), 5))
                for cand in cat_negs:
                    if len(selected_negatives) >= cat_quota: break
                    if cand not in full_history_exclusion and cand != i_id and cand not in selected_negatives:
                        selected_negatives.append(cand)

        # 3. General Negatives (Popularity-based) -> Fill the rest
        # Sample more than needed to account for collisions
        needed = n_negatives - len(selected_negatives)
        if needed > 0:
            pop_candidates = rng.choice(all_items_universe, size=needed * 3)
            for cand in pop_candidates:
                if len(selected_negatives) >= n_negatives: break
                # Deduplicate and Exclude
                if cand not in full_history_exclusion and cand != i_id and cand not in selected_negatives:
                    selected_negatives.append(cand)
        
        # C. Negative Features
        for neg_id in selected_negatives:
            neg_row = fast_compute_features(
                u_id, neg_id, ts, 0.0, 
                current_history, last_ts, 
                similarity_matrix, item_stats_dict, 
                item_categories, user_category_profiles, 
                is_negative=True
            )
            training_data.append(neg_row)
            
        # D. Update State
        if u_id not in user_running_history:
            user_running_history[u_id] = set()
        user_running_history[u_id].add(i_id)
        user_last_ts[u_id] = ts

    df_final = pd.DataFrame(training_data)
    print(f"  Final Dataset: {len(df_final)} rows.")
    return df_final


def save_training_data(training_df: pd.DataFrame, output_path: str):
    training_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    INTERACTIONS_PATH = "data/processed/interactions.csv"
    TRAINING_OUTPUT_PATH = "data/processed/training_data.csv"
    
    if pd.io.common.file_exists(INTERACTIONS_PATH):
        interactions = pd.read_csv(INTERACTIONS_PATH)
        train, test = time_based_split(interactions)
        training_df = build_training_data(train, n_negatives=6)
        save_training_data(training_df, TRAINING_OUTPUT_PATH)
    else:
        print("Interactions file not found.")