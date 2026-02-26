import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Set, Optional, Any

# Note: We use Any for taxonomy_engine to avoid circular imports if strictly typed,
# but ideally you import the class structure if possible.

# -------------------------------------------------------
# Popularity (fallback)
# -------------------------------------------------------
def compute_item_popularity(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total interaction score AND count per item.
    
    Used as fallback for cold-start scenarios.
    
    Args:
        train_df: Training interactions DataFrame
    
    Returns:
        DataFrame with item_id, interaction_score, and interaction_count.
    """
    item_popularity = (
        train_df
        .groupby("item_id")
        .agg(
            interaction_score=("interaction_score", "sum"),
            interaction_count=("interaction_score", "count")
        )
        .reset_index()
        .sort_values("interaction_score", ascending=False)
    )
    return item_popularity


# -------------------------------------------------------
# Item–Item Co-occurrence Matrix
# -------------------------------------------------------
def build_cooccurrence_matrix(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    min_cooccurrence: int = 2,
) -> Dict[int, Dict[int, float]]:
    """
    Build item-item co-occurrence matrix with cosine normalization.
    
    For each user/session:
    1. Take all items interacted with
    2. Generate all item pairs
    3. Count co-occurrences across users
    4. Normalize using cosine similarity
    
    Args:
        interactions: DataFrame with user-item interactions
        user_col: Column name for user IDs
        item_col: Column name for item IDs
        min_cooccurrence: Minimum co-occurrences to include pair (filter noise)
    
    Returns:
        Nested dictionary: {item_id: {similar_item_id: similarity_score}}
    """
    # 1. Generate item pairs per user
    user_items = interactions.groupby(user_col)[item_col].apply(set).to_dict()
    
    # 2. Count co-occurrences
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))
    item_counts = defaultdict(int)
    
    for user_id, items in user_items.items():
        items_list = list(items)
        
        # Count individual item occurrences
        for item in items_list:
            item_counts[item] += 1
        
        # Generate all pairs and count co-occurrences
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                item_a = items_list[i]
                item_b = items_list[j]
                
                # Store both directions for symmetric access
                cooccurrence_counts[item_a][item_b] += 1
                cooccurrence_counts[item_b][item_a] += 1
    
    # 3. Normalize using cosine similarity 
    similarity_matrix = {}
    
    for item_a, cooccurrences in cooccurrence_counts.items():
        similarity_matrix[item_a] = {}
        
        for item_b, count in cooccurrences.items():
            # Filter out low co-occurrence pairs (noise reduction)
            if count < min_cooccurrence:
                continue
            
            # Cosine similarity: count / sqrt(count_a * count_b)
            denominator = np.sqrt(item_counts[item_a] * item_counts[item_b])
            
            if denominator > 0:
                similarity_matrix[item_a][item_b] = count / denominator
    
    return similarity_matrix


# -------------------------------------------------------
# User-level Recommendation (Updated with Taxonomy)
# -------------------------------------------------------
def recommend_item_based(
    user_history: Set[int],
    similarity_matrix: Dict[int, Dict[int, float]],
    top_popular_items: list,
    k: int = 10,
    candidates_per_item: int = 50,
    taxonomy_engine: Optional[Any] = None, # NEW: Optional Taxonomy Engine
) -> list:
    """
    Recommend items using item-item collaborative filtering with explicit cold-start
    and taxonomy-based backfilling.
    
    ALGORITHM:
    1. Collaborative Filtering: Find neighbors for items in user history.
    2. Taxonomy Augmentation (New): If CF candidates < k, find items from same/sibling categories.
    3. Popularity Backfill: If still < k, fill with global popular items.
    
    Args:
        user_history: Set of item IDs the user has interacted with
        similarity_matrix: Pre-computed item-item similarity matrix
        top_popular_items: Pre-sorted list of popular items (for cold-start)
        k: Number of recommendations to return
        candidates_per_item: Number of similar items to consider per history item
        taxonomy_engine: Instance of TaxonomyEngine for category-based retrieval
    
    Returns:
        List of recommended item IDs
    """
    # --- STEP 1: Collaborative Filtering ---
    # Attempt to find items similar to what the user has seen
    candidate_scores = {}
    
    if user_history:
        for item_id in user_history:
            # Get similar items from the matrix
            similar_items_dict = similarity_matrix.get(item_id, {})
            
            if similar_items_dict:
                # Get top candidates for this item
                sorted_similar = sorted(
                    similar_items_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:candidates_per_item]
                
                # Aggregate scores
                for candidate_item, similarity_score in sorted_similar:
                    if candidate_item not in user_history:
                        candidate_scores[candidate_item] = (
                            candidate_scores.get(candidate_item, 0) + similarity_score
                        )
    
    # Sort candidates by aggregated score
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Extract top K item IDs from CF
    recommendations = [item_id for item_id, score in sorted_candidates]
    
    # Trim if we already have enough
    if len(recommendations) > k:
        recommendations = recommendations[:k]
        
    # --- STEP 2: Taxonomy Augmentation (Backfill) ---
    # If CF didn't provide enough items, use the Category Tree to find substitutes
    if len(recommendations) < k and taxonomy_engine is not None and user_history:
        # We need to fill the gap
        current_set = set(recommendations).union(user_history)
        
        # Iterate through user history items to find category-based neighbors
        # (Since set is unordered, this samples arbitrarily from history, which is fine)
        for seed_item in user_history:
            if len(recommendations) >= k:
                break
                
            # Ask taxonomy engine for candidates related to this seed
            # It handles parent/sibling traversal internally
            tax_candidates = taxonomy_engine.get_candidates(
                seed_item_id=seed_item,
                current_candidates=current_set,
                target_k=k # Request enough to fill the quota
            )
            
            # Add unique taxonomy candidates
            for cand in tax_candidates:
                if len(recommendations) >= k:
                    break
                recommendations.append(cand)
                current_set.add(cand)

    # --- STEP 3: Global Popularity (Safety Net) ---
    # If we STILL don't have enough (e.g., cold user or sparse taxonomy), use global popularity
    if len(recommendations) < k:
        popular_backfill = recommend_top_k_popular(
            user_history=user_history.union(set(recommendations)),
            top_popular_items=top_popular_items,
            k=k - len(recommendations)
        )
        recommendations.extend(popular_backfill)
    
    return recommendations


# -------------------------------------------------------
# Popularity Recommendation
# -------------------------------------------------------
def recommend_top_k_popular(
    user_history: Set[int],
    top_popular_items: list,
    k: int = 10,
) -> list:
    """
    Recommend Top-K popular items the user has not interacted with yet.
    """
    recommendations = []
    
    for item in top_popular_items:
        # If user hasn't seen it, add it
        if item not in user_history:
            recommendations.append(item)
        
        # Stop once we have enough recommendations
        if len(recommendations) >= k:
            break
    
    return recommendations