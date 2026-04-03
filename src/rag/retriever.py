import faiss
import numpy as np

class BookRetriever:
    def __init__(self, embeddings, books_df):
        self.books_df = books_df
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]), top_k
        )
        return self.books_df.iloc[indices[0]]