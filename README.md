# 📚 Hybrid Semantic Retrieval and Recommendation System (RAG + CF)
![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-blue)

## 🚀 Overview

This project implements an **end-to-end hybrid recommendation and semantic retrieval system** that combines:

* **Collaborative Filtering (SVD)** for personalization
* **Retrieval-Augmented Generation (RAG)** for semantic search
* **Transformer embeddings (Sentence-BERT)** for understanding natural language queries

The system supports both:

* 👤 User-based recommendations
* 🔍 Natural language search queries

---

## 🎯 Key Features

* 🔹 Personalized recommendations using **Matrix Factorization (SVD)**
* 🔹 Hybrid ranking combining **collaborative filtering + popularity signals**
* 🔹 Semantic search using **Sentence-BERT embeddings**
* 🔹 Fast retrieval with **FAISS vector database**
* 🔹 Query understanding via **LLM-based query expansion**
* 🔹 RAG pipeline for **context-aware retrieval**
* 🔹 Evaluation using **Precision@K and Mean Reciprocal Rank (MRR)**

---

## 🧠 System Architecture

```
                ┌──────────────────────┐
                │     User Input       │
                │  (User ID / Query)   │
                └─────────┬────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
Collaborative Filtering            Semantic Search (RAG)
   (SVD Model)                    (Embeddings + FAISS)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
             Hybrid Ranking System
                       ▼
              Final Recommendations
```

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy, Scikit-learn**
* **SciPy (SVD)**
* **Sentence-Transformers (BERT embeddings)**
* **FAISS (vector search)**
* **TQDM (progress tracking)**

---

## 📂 Project Structure

```
src/
├── data/              # Data loading & preprocessing
├── features/          # User-item matrix
├── models/            # SVD model
├── pipeline/
│   ├── train_pipeline.py
│   ├── recommender.py
│   ├── rag_pipeline.py
├── rag/
│   ├── embedder.py
│   ├── retriever.py
│   ├── generator.py
├── eval/              # Evaluation metrics
main.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python main.py
```

---

## 📊 Model Evaluation

| Model      | Precision@5 | MRR    |
| ---------- | ----------- | ------ |
| Popularity | ~0.007      | ~0.004 |
| SVD        | ~0.011      | ~0.007 |
| Hybrid     | ~0.011      | ~0.007 |

> Note: Performance is constrained by dataset sparsity (Book-Crossing dataset).

---

## 🔍 Example Usage

### 🔹 Recommendation (Collaborative Filtering)

```
🎯 Sample Recommendations:
- Harry Potter
- Lord of the Rings
- Fahrenheit 451
```

### 🔹 Semantic Search (RAG)

```
Query: "dark fantasy books"

🔍 Results:
- Dark-themed fiction
- Fantasy novels
- Adventure-based stories
```

---

## 🧪 Key Learnings

* Handling **sparse user-item matrices**
* Importance of **ranking-based evaluation metrics**
* Combining **symbolic (CF) + semantic (RAG)** approaches
* Improving retrieval using **query expansion & metadata enrichment**
* Debugging real-world ML systems with **noisy data**

---

## 🚧 Limitations

* Dataset lacks **rich metadata (genres/descriptions)**
* Exact-match evaluation underestimates semantic relevance
* SVD struggles with **cold-start users/items**

---

## 🚀 Future Improvements

* 🔹 Replace SVD with **ALS (implicit feedback)**
* 🔹 Use **book descriptions / summaries** for better embeddings
* 🔹 Add **FastAPI / Streamlit UI**
* 🔹 Deploy as a **real-time recommendation API**
* 🔹 Fine-tune embeddings for domain-specific retrieval

---

## 👨‍💻 Author

**Abhay Raj Singh**

---

## ⭐ If you like this project, give it a star!
