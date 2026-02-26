"""
TAXONOMY.PY
-----------
Purpose: Implements a Taxonomy-Augmented Candidate Generator using item properties 
and category tree structures.

Role in Pipeline:
    - Serves as a "Backfill Layer" for Candidate Generation.
    - When Item-Item CF returns too few candidates (sparsity), this module 
      uses the category tree to find relevant substitutes.

Data Sources:
    1. category_tree.csv: Hierarchical structure (child -> parent).
    2. item_properties_part1/2.csv: Maps item_id -> category_id.
    3. interactions.csv: Used to determine "category-level popularity".
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Set, Optional, Tuple

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORY_TREE_PATH = os.path.join(BASE_DIR, "data", "raw", "category_tree.csv")
ITEM_PROPS_P1_PATH = os.path.join(BASE_DIR, "data", "raw", "item_properties_part1.csv")
ITEM_PROPS_P2_PATH = os.path.join(BASE_DIR, "data", "raw", "item_properties_part2.csv")


class TaxonomyEngine:
    """
    A context-aware engine that retrieves candidates based on catalog hierarchy.
    
    It pre-computes two critical artifacts:
    1. Category Tree: Fast traversal from Child -> Parent -> Sibling.
    2. Category Popularity: Lists of top items per category, sorted by interaction volume.
    """

    def __init__(self, interactions: pd.DataFrame):
        """
        Initialize the engine by loading raw metadata and computing category stats.
        
        Args:
            interactions: DataFrame containing [user_id, item_id, interaction_score].
                          Used to rank items within categories.
        """
        print("Initializing Taxonomy Engine...")
        
        # 1. Load Item -> Category Mapping
        self.item_to_category = self._load_item_categories()
        print(f"  Mapped {len(self.item_to_category):,} items to categories.")
        
        # 2. Load Category Tree Structure
        # self.parents: category_id -> parent_id
        # self.children: category_id -> list of child_ids
        self.parents, self.children = self._load_category_tree()
        print(f"  Loaded category tree with {len(self.parents)} relationships.")
        
        # 3. Pre-compute Category Popularity
        # self.category_top_items: category_id -> [item_id1, item_id2, ...] (sorted desc)
        self.category_top_items = self._compute_category_popularity(interactions)
        print(f"  Computed popularity for {len(self.category_top_items):,} categories.")
        
    def _load_item_categories(self) -> Dict[int, int]:
        """
        Parses raw item property files to extract 'categoryid' for each item.
        Handles both part1 and part2 files.
        """
        mapping = {}
        
        for path in [ITEM_PROPS_P1_PATH, ITEM_PROPS_P2_PATH]:
            if not os.path.exists(path):
                print(f"  [WARNING] File not found: {path}. Skipping.")
                continue
                
            # Reading chunked to handle potential large files, though strict memory 
            # management isn't implemented here for simplicity.
            # We filter for property == 'categoryid' immediately.
            try:
                df = pd.read_csv(path)
                
                # Filter for categoryid property
                # Based on provided snippet, column is 'property' and value is 'categoryid'
                cat_df = df[df['property'] == 'categoryid'].copy()
                
                # Values need to be integers
                # Some values might be dirty, so we coerce errors
                cat_df['value'] = pd.to_numeric(cat_df['value'], errors='coerce')
                cat_df = cat_df.dropna(subset=['value'])
                
                # Update mapping
                # Using dict() construction is faster than iterating rows
                current_map = dict(zip(cat_df['itemid'], cat_df['value'].astype(int)))
                mapping.update(current_map)
            except Exception as e:
                 print(f"  [ERROR] Could not process {path}: {e}")
            
        return mapping

    def _load_category_tree(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Loads the parent-child relationships.
        
        Returns:
            parents: Dict[child_id, parent_id]
            children: Dict[parent_id, List[child_id]]
        """
        if not os.path.exists(CATEGORY_TREE_PATH):
            print("  [WARNING] Category tree file not found.")
            return {}, {}
            
        df = pd.read_csv(CATEGORY_TREE_PATH)
        # Expected cols: categoryid, parentid
        
        parents = {}
        children = {}
        
        for row in df.itertuples():
            cid = row.categoryid
            pid = row.parentid
            
            # parentid can be NaN (root nodes)
            if pd.notnull(pid):
                pid = int(pid)
                parents[cid] = pid
                
                if pid not in children:
                    children[pid] = []
                children[pid].append(cid)
                
        return parents, children

    def _compute_category_popularity(self, interactions: pd.DataFrame) -> Dict[int, List[int]]:
        """
        Generates a sorted list of popular items for every category.
        
        Logic:
        1. Calculate global item popularity (interaction_score sum).
        2. Assign items to categories using self.item_to_category.
        3. Group by category and sort items descending.
        """
        # 1. Global Item Popularity
        item_scores = (
            interactions
            .groupby("item_id")["interaction_score"]
            .sum()
            .reset_index(name="score")
        )
        
        # 2. Map to Category
        # We map items to their category. Items without a category are dropped from this index.
        item_scores["category_id"] = item_scores["item_id"].map(self.item_to_category)
        
        # --- FIX: Added .copy() to prevent SettingWithCopyWarning ---
        valid_items = item_scores.dropna(subset=["category_id"]).copy()
        valid_items["category_id"] = valid_items["category_id"].astype(int)
        
        # 3. Group and Sort
        # Result: {cat_id: [item_A, item_B, ...]}
        cat_pop_map = {}
        
        # Sort by category and score (desc) to facilitate fast grouping
        valid_items = valid_items.sort_values(["category_id", "score"], ascending=[True, False])
        
        # Groupby is efficient on sorted data
        # Aggregate item_ids into a list for each category
        grouped = valid_items.groupby("category_id")["item_id"].apply(list)
        cat_pop_map = grouped.to_dict()
        
        return cat_pop_map

    def get_candidates(
        self, 
        seed_item_id: int, 
        current_candidates: Set[int], 
        target_k: int = 50
    ) -> List[int]:
        """
        Augment a candidate set using taxonomy traversals.
        
        ALGORITHM:
        1. Check if we need more candidates (len(current) < target_k).
        2. Level 1 (Same Category): Fetch popular items from the seed item's direct category.
        3. Level 2 (Siblings/Parent): If still insufficient, fetch from parent category and sibling categories.
        4. Level 3 (Grandparent): If still insufficient, traverse one level higher.
        
        Args:
            seed_item_id: The item the user interacted with (basis for expansion).
            current_candidates: Set of item IDs already retrieved (e.g., by CF).
            target_k: Total number of candidates desired.
            
        Returns:
            List of NEW item IDs to add (excludes items already in current_candidates).
        """
        new_candidates = []
        
        # 1. Identify Seed Category
        seed_cat = self.item_to_category.get(seed_item_id)
        if seed_cat is None:
            # Seed item has no known category, cannot expand via taxonomy
            return []
            
        # Strategy: Define layers of categories to inspect
        # Layer 0: The category itself
        categories_to_check = [seed_cat]
        
        # Layer 1: Parent and Siblings
        parent = self.parents.get(seed_cat)
        if parent:
            # Add parent itself
            categories_to_check.append(parent)
            # Add siblings (other children of the parent)
            siblings = self.children.get(parent, [])
            categories_to_check.extend([sib for sib in siblings if sib != seed_cat])
            
            # Layer 2: Grandparent (Optional deep fallback)
            grandparent = self.parents.get(parent)
            if grandparent:
                categories_to_check.append(grandparent)
                
        # 2. Harvest Candidates
        # We iterate through prioritized categories and fill the bucket
        for cat_id in categories_to_check:
            # Stop if we hit the target
            if len(current_candidates) + len(new_candidates) >= target_k:
                break
                
            # Get popular items in this category
            popular_in_cat = self.category_top_items.get(cat_id, [])
            
            for item in popular_in_cat:
                if len(current_candidates) + len(new_candidates) >= target_k:
                    break
                
                # Exclude seed and already selected items
                if item != seed_item_id and item not in current_candidates and item not in new_candidates:
                    new_candidates.append(item)
                    
        return new_candidates