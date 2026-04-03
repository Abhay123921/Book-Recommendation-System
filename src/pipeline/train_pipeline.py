from src.data.load_data import load_data
from src.data.preprocess import (
    clean_column_names,
    preprocess_data,
    split_data_userwise,
    filter_train_data
)
from src.features.build_matrix import build_matrix
from src.models.svd import train_svd


def run_training():

    # -------------------------------
    # Step 1: Load data
    # -------------------------------
    books, ratings, users = load_data()

    # -------------------------------
    # Step 2: Clean column names
    # -------------------------------
    books, users, ratings = clean_column_names(books, users, ratings)

    # -------------------------------
    # Step 3: Basic preprocessing
    # -------------------------------
    ratings = preprocess_data(ratings)

    # -------------------------------
    # Step 4: Train-test split
    # -------------------------------
    train_ratings, test_ratings = split_data_userwise(ratings)

    # -------------------------------
    # Step 5: Filter ONLY train
    # -------------------------------
    train_ratings = filter_train_data(train_ratings)

    # -------------------------------
    # Step 6: Build matrix
    # -------------------------------
    matrix = build_matrix(train_ratings)

    # -------------------------------
    # Step 7: Train model
    # -------------------------------
    preds = train_svd(matrix, k=100)

    return books, matrix, preds, train_ratings, test_ratings