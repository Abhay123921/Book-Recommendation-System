from src.data.load_data import load_data
from src.data.preprocess import clean_column_names, preprocess_data
from src.features.build_matrix import build_matrix

# STEP 1: Load data
books, ratings, users = load_data()

# STEP 2: Clean column names
books, users, ratings = clean_column_names(books, users, ratings)

# STEP 3: Preprocess
ratings = preprocess_data(books, ratings)

# STEP 4: Build matrix
matrix = build_matrix(ratings)

# Debug output
print("Matrix shape:", matrix.shape)
print(matrix.head())