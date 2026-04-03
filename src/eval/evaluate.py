from tqdm import tqdm
import numpy as np


def evaluate_model(matrix, test_ratings, recommend_fn, k=5):

    precision_scores = []
    reciprocal_ranks = []

    # Pre-group test data (🔥 faster)
    test_grouped = test_ratings.groupby('user_id')['isbn'].apply(list).to_dict()

    users = list(test_grouped.keys())

    for user in tqdm(users):

        # Skip users not in training
        if user not in matrix.index:
            continue

        # Skip low-activity users
        if (matrix.loc[user] > 0).sum() < 5:
            continue

        actual_items = test_grouped.get(user, [])

        if not actual_items:
            continue

        # Leave-one-out → single item
        actual_item = actual_items[0]

        # Get predictions
        predicted_items = recommend_fn(user)

        if not predicted_items:
            continue

        # -------------------------------
        # Precision@K
        # -------------------------------
        hit = 1 if actual_item in predicted_items[:k] else 0
        precision_scores.append(hit)

        # -------------------------------
        # MRR
        # -------------------------------
        rr = 0
        for i, item in enumerate(predicted_items):
            if item == actual_item:
                rr = 1 / (i + 1)
                break

        reciprocal_ranks.append(rr)

    # -------------------------------
    # Final metrics
    # -------------------------------
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    return avg_precision, mrr