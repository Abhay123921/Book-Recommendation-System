import pandas as pd


def build_matrix(ratings):

    # -------------------------------
    # Step 1: Normalize ratings (per user)
    # -------------------------------
    ratings['rating_norm'] = ratings.groupby('user_id')['rating'].transform(
        lambda x: x - x.mean()
    )

    # -------------------------------
    # Step 2: Pivot table
    # -------------------------------
    user_item_matrix = ratings.pivot_table(
        index='user_id',
        columns='isbn',
        values='rating_norm'
    )

    # -------------------------------
    # Step 3: Fill missing with 0
    # -------------------------------
    user_item_matrix = user_item_matrix.fillna(0)

    return user_item_matrix