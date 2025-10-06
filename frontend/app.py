import streamlit as st
import requests
import io

API_URL = "http://127.0.0.1:8000/search-similarity"

st.set_page_config(page_title="Parallel Document Similarity Search", layout="wide")
st.title("📄 Parallel Document Similarity Search")
st.markdown("Upload a document to find the most similar documents in the dataset.")

# File uploader (accept any file type)
uploaded_file = st.file_uploader("📤 Choose a document file", type=None)

# Slider for number of top results
n_results = st.slider("Number of Top Similar Documents", 1, 10, 5)

if uploaded_file is not None:
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")

    # Preview if text file
    if uploaded_file.type.startswith("text/"):
        st.subheader("📘 Preview of Uploaded Document")
        try:
            preview_text = uploaded_file.getvalue().decode("utf-8")[:500]
            st.text(preview_text + "..." if len(preview_text) == 500 else preview_text)
        except Exception:
            st.warning("Unable to display preview for this file type.")

    # Search button
    if st.button("🔍 Start Parallel Similarity Search"):
        with st.spinner("Running parallel cosine similarity computation..."):
            try:
                # Create BytesIO stream for file transfer
                file_content_io = io.BytesIO(uploaded_file.getvalue())
                files = {
                    "query_file": (
                        uploaded_file.name,
                        file_content_io,
                        uploaded_file.type or "application/octet-stream"
                    )
                }
                params = {"n_results": n_results}

                # Send request to FastAPI backend
                response = requests.post(API_URL, files=files, params=params)
                response.raise_for_status()
                results = response.json()

                # Display results
                st.subheader("🏆 Top Matching Documents")
                if isinstance(results, list) and len(results) > 0:
                    for i, result in enumerate(results, start=1):
                        score = result.get("score", 0.0)
                        doc_id = result.get("document_id", "Unknown")
                        preview = result.get("document_preview", "No preview available.")

                        st.markdown(f"**{i}. Document ID:** `{doc_id}` | **Score:** `{score:.4f}`")
                        st.code(preview, language="markdown")
                else:
                    st.warning("No similar documents found in the dataset.")

            except requests.exceptions.ConnectionError:
                st.error(f"🚫 Unable to connect to the backend API at `{API_URL}`. Please start FastAPI server first.")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"❗HTTP Error: {http_err}")
                try:
                    st.code(response.json(), language="json")
                except Exception:
                    st.code(response.text, language="text")
            except Exception as e:
                st.error(f"⚠️ Unexpected Error: {str(e)}")
else:
    st.info("👆 Please upload a file to begin similarity search.")
