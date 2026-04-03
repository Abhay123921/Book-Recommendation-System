def recommend_hybrid(user_id, matrix, preds, ratings, top_n=10, alpha=0.98):

    # -------------------------------
    # Step 0: Cold-start
    # -------------------------------
    popularity_series = ratings.groupby('isbn')['rating'].count()

    if user_id not in matrix.index:
        return popularity_series.sort_values(ascending=False).index[:top_n].tolist()

    # -------------------------------
    # Step 1: User index
    # -------------------------------
    user_idx = matrix.index.get_loc(user_id)

    # -------------------------------
    # Step 2: SVD scores (normalized)
    # -------------------------------
    svd_scores = preds[user_idx, :]

    svd_min, svd_max = svd_scores.min(), svd_scores.max()
    if svd_max - svd_min > 0:
        svd_scores = (svd_scores - svd_min) / (svd_max - svd_min)
    else:
        svd_scores = np.zeros_like(svd_scores)

    # -------------------------------
    # Step 3: Popularity scores (aligned)
    # -------------------------------
    popularity_scores = (
        popularity_series
        .reindex(matrix.columns)
        .fillna(0)
        .values
    )

    pop_max = popularity_scores.max()
    if pop_max > 0:
        popularity_scores = popularity_scores / pop_max

    # -------------------------------
    # Step 4: Hybrid score
    # -------------------------------
    final_scores = alpha * svd_scores + (1 - alpha) * popularity_scores

    # -------------------------------
    # Step 5: Remove seen items
    # -------------------------------
    user_interactions = matrix.loc[user_id]
    seen_items = set(user_interactions[user_interactions > 0].index)

    # -------------------------------
# Step 6: Rank ALL items (NO restriction)
# -------------------------------
    top_items = final_scores.argsort()[::-1]

# -------------------------------
# Step 7: Filter unseen + top-N
# -------------------------------
    recs = []
    for idx in top_items:
        isbn = matrix.columns[idx]

        if isbn not in seen_items:
            recs.append(isbn)

        if len(recs) >= top_n:
            break

    return recs
def get_book_details(isbns, books_df):
    df = books_df[books_df['isbn'].isin(isbns)][['isbn', 'title', 'author']]
    return df.set_index('isbn').loc[isbns].reset_index()