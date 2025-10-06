import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from dask.distributed import Client, LocalCluster
import dask.array as da

# --- 1️⃣ Initialize FastAPI app ---
app = FastAPI(title="Parallel Document Similarity Search API")

# --- 2️⃣ Initialize Dask Cluster ---
cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)

# --- 3️⃣ Load and preprocess dataset ---
def load_and_preprocess(data_path):
    """Loads documents and applies basic preprocessing."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus

# --- 4️⃣ TF-IDF Vectorization ---
def get_tfidf_vectors(corpus, query_doc=None, vectorizer=None):
    """Fits/transforms corpus and transforms query into TF-IDF vectors."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english')
        corpus_vectors = vectorizer.fit_transform(corpus)
    else:
        corpus_vectors = vectorizer.transform(corpus)

    if query_doc:
        query_vector = vectorizer.transform([query_doc])
        return corpus_vectors, query_vector, vectorizer
    
    return corpus_vectors, None, vectorizer

# --- 5️⃣ Parallel Cosine Similarity using Dask ---
def parallel_cosine_similarity(corpus_vectors, query_vector, n_top=5):
    """Calculates cosine similarity in parallel using Dask."""
    X_da = da.from_array(corpus_vectors.toarray(), chunks=(1000, corpus_vectors.shape[1]))
    Y_da = da.from_array(query_vector.toarray(), chunks=(1, query_vector.shape[1]))
    
    dot_product = da.dot(X_da, Y_da.T)
    corpus_norms = da.linalg.norm(X_da, axis=1)
    query_norm = da.linalg.norm(Y_da, axis=1)[0]
    
    similarity_da = (dot_product.T / (corpus_norms * query_norm))
    similarity_scores = similarity_da.compute().flatten()
    
    top_indices = np.argsort(similarity_scores)[::-1][:n_top]
    top_scores = similarity_scores[top_indices]
    
    return list(zip(top_indices.tolist(), top_scores.tolist()))

# --- 6️⃣ Load Dataset once on startup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "dataset.txt")  # expects dataset inside "data" folder
CORPUS = load_and_preprocess(DATA_PATH)
CORPUS_VECTORS, _, VECTORIZER = get_tfidf_vectors(CORPUS)

# --- 7️⃣ FastAPI Endpoint ---
@app.post("/search-similarity")
async def search_similarity(query_file: UploadFile = File(...), n_results: int = Query(5, ge=1, le=10)):
    """
    Accepts a text file from Streamlit frontend and returns top N similar documents.
    """
    try:
        query_text = (await query_file.read()).decode("utf-8").strip()
        if not query_text:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        _, query_vector, _ = get_tfidf_vectors(CORPUS, query_text, VECTORIZER)
        top_matches = parallel_cosine_similarity(CORPUS_VECTORS, query_vector, n_top=n_results)

        results = [
            {
                "document_id": idx,
                "score": float(score),
                "document_preview": CORPUS[idx][:200]
            }
            for idx, score in top_matches
        ]
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 8️⃣ Health Check Route ---
@app.get("/")
def root():
    return {"message": "Parallel Document Similarity Search API is running!"}
