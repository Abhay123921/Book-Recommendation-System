import pandas as pd

def load_data():
    # Load books
    books = pd.read_csv(
        "data/raw/books.csv",
        sep=';',
        encoding='latin-1'
    )

    # Load users
    users = pd.read_csv(
        "data/raw/users.csv",
        sep=';',
        dtype={'User-ID': str},
        encoding='latin-1',
        low_memory=False
    )

    # Load ratings
    ratings = pd.read_csv(
        "data/raw/ratings.csv",
        sep=';',
        dtype={'User-ID': str, 'ISBN': str},
        encoding='latin-1'
    )

    # -------------------------------
    # Clean column names (IMPORTANT)
    # -------------------------------
    books.columns = books.columns.str.strip()
    users.columns = users.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    # -------------------------------
    # Ensure ISBN is string everywhere
    # -------------------------------
    books['ISBN'] = books['ISBN'].astype(str)

    return books, ratings, users