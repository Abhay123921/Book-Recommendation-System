from src.pipeline.train_pipeline import run_training
from src.eval.evaluate import evaluate_model
from src.pipeline.recommender import recommend_hybrid, get_book_details
from src.pipeline.rag_pipeline import RAGPipeline


if __name__ == "__main__":

    print("🚀 Starting pipeline...")

    # -------------------------------
    # Run full training pipeline
    # -------------------------------
    books, matrix, preds, train_ratings, test_ratings = run_training()

    print("✅ Model trained")

    # -------------------------------
    # Initialize RAG
    # -------------------------------
    rag = RAGPipeline(books)

    # -------------------------------
    # Recommendation wrapper
    # -------------------------------
    def recommend_wrapper(user_id):
        return recommend_hybrid(user_id, matrix, preds, train_ratings)

    # -------------------------------
    # Baseline Models
    # -------------------------------
    def popularity_recommend(user_id):
        popular = (
            train_ratings.groupby('isbn')['rating']
            .count()
            .sort_values(ascending=False)
        )
        return popular.index[:10].tolist()

    def svd_recommend(user_id):
        if user_id not in matrix.index:
            return popularity_recommend(user_id)

        user_idx = matrix.index.get_loc(user_id)
        scores = preds[user_idx]

        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        user_interactions = matrix.loc[user_id]
        seen_items = set(user_interactions[user_interactions > 0].index)

        ranked = scores.argsort()[::-1]

        recs = []
        for idx in ranked:
            isbn = matrix.columns[idx]
            if isbn not in seen_items:
                recs.append(isbn)
            if len(recs) >= 10:
                break

        return recs

    # -------------------------------
    # Evaluate main model
    # -------------------------------
    precision, mrr = evaluate_model(
        matrix,
        test_ratings,
        recommend_wrapper,
        k=5
    )

    print(f"📊 Precision@5: {precision:.3f}")
    print(f"📊 MRR: {mrr:.3f}")

    # -------------------------------
    # Compare models
    # -------------------------------
    print("\n📊 Evaluating Models...\n")

    pop_precision, pop_mrr = evaluate_model(
        matrix,
        test_ratings,
        popularity_recommend
    )

    print(f"📊 Popularity → Precision@5: {pop_precision:.3f}, MRR: {pop_mrr:.3f}")

    svd_precision, svd_mrr = evaluate_model(
        matrix,
        test_ratings,
        svd_recommend
    )

    print(f"📊 SVD → Precision@5: {svd_precision:.3f}, MRR: {svd_mrr:.3f}")

    hyb_precision, hyb_mrr = evaluate_model(
        matrix,
        test_ratings,
        recommend_wrapper
    )

    print(f"📊 Hybrid → Precision@5: {hyb_precision:.3f}, MRR: {hyb_mrr:.3f}")

    # -------------------------------
    # Sample CF recommendation
    # -------------------------------
    sample_user = matrix.index[0]

    recs = recommend_wrapper(sample_user)
    book_details = get_book_details(recs, books)

    print("\n🎯 Sample Recommendations:")
    print(book_details)

    # -------------------------------
    # RAG Search
    # -------------------------------
    query = "dark fantasy books with complex characters"

    rag_results = rag.search(query, top_k=5)

    print("\n🔍 RAG Results:")
    print(rag_results)