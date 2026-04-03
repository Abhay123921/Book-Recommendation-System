from sklearn.metrics.pairwise import cosine_similarity

def user_cf(matrix, user_id, k=10):

    similarity = cosine_similarity(matrix)

    user_index = matrix.index.get_loc(user_id)
    sim_scores = similarity[user_index]

    similar_users = sim_scores.argsort()[::-1][1:k+1]

    return similar_users
