import os
import sys
import requests
import faiss
import numpy as np
import pickle
import hashlib
from dotenv import load_dotenv

# --- Cloud and Local Imports ---
import vertexai
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
sys.path.append(os.path.join(os.path.dirname(__file__), 'ingestion'))
from text_processor import clean_gutenberg_text, chunk_text_by_paragraph

def build_index_for_book(book_id: str):
    """
    Downloads a book, generates embeddings using a cache, and saves a local FAISS index.
    """
    print(f"--- Building local FAISS index for Book ID: {book_id} ---")

    # --- Initialize Clients ---
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    bq_client = bigquery.Client()
    cache_table_id = f"{project_id}.scribe_v3_dataset.embedding_cache"
    
    # --- Process the book ---
    print("-> Downloading and chunking book...")
    book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(book_url, timeout=15)
    cleaned_text = clean_gutenberg_text(response.text)
    chunks = chunk_text_by_paragraph(cleaned_text)
    print(f"-> Found {len(chunks)} chunks.")

    # --- Generate Embeddings Using Cache ---
    print("-> Getting embeddings, using cache...")
    embedding_vectors = []
    api_calls = 0
    BATCH_SIZE = 100

    for i in range(0, len(chunks), BATCH_SIZE):
        chunk_batch = chunks[i:i + BATCH_SIZE]
        
        hashes_to_check = [hashlib.sha256(text.encode()).hexdigest() for text in chunk_batch]
        query = f"SELECT chunk_hash, embedding FROM `{cache_table_id}` WHERE chunk_hash IN UNNEST(@hashes)"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hashes", "STRING", hashes_to_check)]
        )
        cached_results = {row.chunk_hash: row.embedding for row in bq_client.query(query, job_config=job_config)}
        
        new_texts_for_batch = []
        hash_to_text_map = {}
        for text, text_hash in zip(chunk_batch, hashes_to_check):
            if text_hash not in cached_results:
                new_texts_for_batch.append(text)
                hash_to_text_map[text_hash] = text
        
        if new_texts_for_batch:
            api_calls += len(new_texts_for_batch)
            embedding_responses = embedding_model.get_embeddings(new_texts_for_batch)
            
            rows_to_insert = []
            for text_hash, embedding in zip(hash_to_text_map.keys(), embedding_responses):
                embedding_vector = embedding.values
                cached_results[text_hash] = embedding_vector
                rows_to_insert.append({"chunk_hash": text_hash, "embedding": embedding_vector})
            
            errors = bq_client.insert_rows_json(cache_table_id, rows_to_insert)
            if errors: print(f"Error inserting into cache: {errors}")

        for text_hash in hashes_to_check:
            embedding_vectors.append(cached_results[text_hash])

    print(f"-> Embeddings retrieved. Made {api_calls} new API calls.")
    
    # --- Build and Save FAISS Index ---
    db_vectors = np.array(embedding_vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(768)
    index.add(db_vectors)
    faiss.write_index(index, f"app/local_index_{book_id}.faiss")
    print(f"-> FAISS index saved to 'app/local_index_{book_id}.faiss'")

    # --- Save Metadata (chunk IDs and text) ---
    chunk_ids = [f"{book_id}-{i}" for i in range(len(chunks))]
    chunk_id_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
    chunk_text_map = {chunk_id: text for chunk_id, text in zip(chunk_ids, chunks)}

    with open(f"app/local_metadata_{book_id}.pkl", 'wb') as f:
        pickle.dump({"id_map": chunk_id_map, "text_map": chunk_text_map}, f)
    print(f"-> Metadata and text chunks saved to 'app/local_metadata_{book_id}.pkl'")
    
    print("--- Build complete! ---")

if __name__ == "__main__":
    build_index_for_book("2680")