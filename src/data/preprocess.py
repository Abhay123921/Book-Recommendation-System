import pandas as pd


# -------------------------------
# Clean column names
# -------------------------------
def clean_column_names(books, users, ratings):

    books = books.rename(columns={
        "ISBN": "isbn",
        "Title": "title",
        "Author": "author",
        "Year": "year",
        "Publisher": "publisher"
    })

    users = users.rename(columns={
        "User-ID": "user_id",
        "Age": "age"
    })

    ratings = ratings.rename(columns={
        "User-ID": "user_id",
        "ISBN": "isbn",
        "Rating": "rating"
    })

    return books, users, ratings


# -------------------------------
# Preprocess (LIGHT filtering only)
# -------------------------------
def preprocess_data(ratings):

    # Remove invalid ratings
    ratings = ratings[ratings['rating'] > 0]

    return ratings


# -------------------------------
# User-wise split (leave-one-out)
# -------------------------------
def split_data_userwise(ratings):
    train_list = []
    test_list = []

    for user, group in ratings.groupby('user_id'):

        # Need at least 2 interactions
        if len(group) < 2:
            continue

        # Shuffle interactions
        group = group.sample(frac=1, random_state=42)

        # Last item → test
        train_list.append(group.iloc[:-1])
        test_list.append(group.iloc[-1:])

    train = pd.concat(train_list)
    test = pd.concat(test_list)

    return train, test


# -------------------------------
# Filter sparse data (ONLY TRAIN)
# -------------------------------
def filter_train_data(train):

    # Filter users
    user_counts = train['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    train = train[train['user_id'].isin(valid_users)]

    # Filter books
    book_counts = train['isbn'].value_counts()
    valid_books = book_counts[book_counts >= 5].index
    train = train[train['isbn'].isin(valid_books)]

    return train