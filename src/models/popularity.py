def popular_books(ratings, top_n=10):
    return (
        ratings.groupby('isbn')['rating']
        .count()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )