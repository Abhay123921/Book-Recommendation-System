from src.rag.embedder import embed_text
from src.rag.retriever import BookRetriever
from src.rag.generator import expand_query


class RAGPipeline:
    def __init__(self, books_df):
        self.books_df = books_df

        # -------------------------------
        # 🔥 ADD THIS FUNCTION HERE
        # -------------------------------
        def enrich_book_text(row):
            title = (row['title'] or "").lower()

            text = f"Book: {title}. "

    # --- Domain filtering ---
            if any(word in title for word in ["analysis", "mathematics", "physics", "guide", "manual"]):
                text += "non-fiction, academic, technical"
            else:
                text += "fiction, novel"

    # --- Genre heuristics ---
            if any(word in title for word in ["harry potter", "wizard", "magic"]):
                text += ", fantasy, magic, young adult"

            elif any(word in title for word in ["lord of the rings", "hobbit"]):
                text += ", epic fantasy, adventure, dark fantasy"

            elif "dark" in title:
                text += ", horror, dark fiction, thriller"

            else:
                text += ", general fiction"

            return text

        # -------------------------------
        # 🔥 REPLACE OLD TEXT CREATION
        # -------------------------------
        texts = books_df.apply(enrich_book_text, axis=1).tolist()

        # -------------------------------
        # Embeddings
        # -------------------------------
        embeddings = embed_text(texts)

        # -------------------------------
        # FAISS index
        # -------------------------------
        self.retriever = BookRetriever(embeddings, books_df)

    def search(self, query, top_k):

        expanded_query = expand_query(query)

        query_embedding = embed_text([expanded_query])[0]

        results = self.retriever.search(query_embedding, top_k=20)

        results = results[
        ~results['title'].str.lower().str.contains(
        "analysis|mathematics|guide|manual", na=False
        )
        ]

        return results[['isbn', 'title', 'author']]