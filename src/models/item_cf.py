def item_cf(matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    item_sim = cosine_similarity(matrix.T)
    return item_sim