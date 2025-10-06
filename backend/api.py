from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dask.distributed import Client, LocalCluster
import dask.array as da
import os

# --- Initialize Dask Cluster ---
cluster = LocalCluster(n_workers=4, threads_per_worker=1)
dask_client = Client(cluster)

app = FastAPI(title="Parallel Document Similarity API")

# Allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/dataset.txt"
os.makedirs("data", exist_ok=True)

# ---------- Load + Vectorize ----------
def load_and_preprocess(data_path):
    if not os.path.exists(data_path):
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus


def get_tfidf_vectors(corpus):
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus_vectors = vectorizer.fit_transform(corpus)
    return corpus_vectors, vectorizer


def parallel_cosine_similarity(corpus_vectors, query_vector, n_top=5):
    X = corpus_vectors.toarray()
    Y = query_vector.toarray()

    X_da = da.from_array(X, chunks=(min(1000, X.shape[0]), X.shape[1]))
    Y_da = da.from_array(Y, chunks=(1, Y.shape[1]))

    dot_product = da.dot(X_da, Y_da.T)
    corpus_norms = da.linalg.norm(X_da, axis=1)
    query_norm = da.linalg.norm(Y_da, axis=1)[0]

    similarity_da = dot_product.T / (corpus_norms * query_norm)
    similarity_scores = similarity_da.compute().flatten()

    # Replace NaNs with zeros (to avoid JSON errors)
    similarity_scores = np.nan_to_num(similarity_scores, nan=0.0)

    top_indices = np.argsort(similarity_scores)[::-1][:n_top]
    top_scores = similarity_scores[top_indices]
    return list(zip(top_indices.tolist(), top_scores.tolist()))


# ---------- API Endpoint ----------
@app.post("/search-similarity")
async def search_similarity(query_file: UploadFile, n_results: int = 5):
    try:
        # Read uploaded content safely
        query_content = (await query_file.read()).decode("utf-8", errors="ignore").strip()
        if not query_content:
            return {"error": "Uploaded file is empty."}

        # Append query to dataset
        with open(DATA_PATH, "a", encoding="utf-8") as f:
            f.write("\n" + query_content)

        corpus = load_and_preprocess(DATA_PATH)
        if len(corpus) == 0:
            return {"error": "Dataset is empty. Please add more documents first."}

        corpus_vectors, vectorizer = get_tfidf_vectors(corpus)
        query_vector = vectorizer.transform([query_content])

        results = parallel_cosine_similarity(corpus_vectors, query_vector, n_top=n_results + 1)

        response = []
        for idx, score in results:
            if corpus[idx] == query_content:
                continue
            response.append({
                "document_id": idx,
                "score": round(float(score), 4),
                "document_preview": corpus[idx][:300] + "..." if len(corpus[idx]) > 300 else corpus[idx]
            })
            if len(response) >= n_results:
                break

        return response

    except Exception as e:
        return {"error": str(e)}
