import numpy as np


# -------------------------------
# Precision@K (leave-one-out)
# -------------------------------
def precision_at_k(actual_item, predicted, k=5):
    return 1 if actual_item in predicted[:k] else 0


# -------------------------------
# Reciprocal Rank
# -------------------------------
def reciprocal_rank(actual_item, predicted):
    for i, item in enumerate(predicted):
        if item == actual_item:
            return 1 / (i + 1)
    return 0