import os
import time
import requests
import json
import uuid
import hashlib
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv

# --- Vertex AI & BigQuery Imports ---
import vertexai
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from ingestion.text_processor import clean_gutenberg_text, chunk_text_by_paragraph

def get_book_metadata(book_id):
    """Fetches book metadata from the Gutendex API."""
    try:
        url = f"http://gutendex.com/books/{book_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for book {book_id} from Gutendex: {e}")
        return None

def map_gutenberg_to_schema(book_metadata, chunk_text, chunk_index):
    """Maps Gutenberg data to the SCRIBE metadata schema."""
    book_id = book_metadata.get("id")
    authors = book_metadata.get("authors", [])
    author_names = [author["name"] for author in authors]
    publication_date = authors[0].get("death_year") if authors else None

    mapped_data = {
        "id": str(uuid.uuid4()),
        "source_id": f"gutenberg:{book_id}",
        "title": book_metadata.get("title"),
        "author": author_names,
        "publisher": "Project Gutenberg",
        "publication_date": publication_date,
        "source_url": f"https://www.gutenberg.org/ebooks/{book_id}",
        "language": book_metadata.get("languages", ["en"])[0],
        "canonical_reference": f"{book_metadata.get('title')} - Chunk {chunk_index + 1}",
        "text_type": "book",
        "chunk_level": "paragraph",
        "text": chunk_text
    }
    return {k: v for k, v in mapped_data.items() if v is not None}

def load_book_ids_from_manifest(manifest_path):
    """Loads book IDs from a text manifest file."""
    try:
        with open(manifest_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return []

def get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, chunks):
    """Gets embeddings for a list of text chunks, using a BigQuery cache."""
    print(f"  -> Getting embeddings for {len(chunks)} chunks, using BigQuery cache...")
    embedding_vectors = []
    api_calls = 0
    BATCH_SIZE = 50 # Recommended batch size for the API

    for i in range(0, len(chunks), BATCH_SIZE):
        chunk_batch = chunks[i:i + BATCH_SIZE]
        hashes_to_check = [hashlib.sha256(text.encode()).hexdigest() for text in chunk_batch]
        
        query = f"SELECT chunk_hash, embedding FROM `{cache_table_id}` WHERE chunk_hash IN UNNEST(@hashes)"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("hashes", "STRING", hashes_to_check)
            ]
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
            inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in new_texts_for_batch]
            embedding_responses = embedding_model.get_embeddings(inputs)
            
            rows_to_insert = []
            for text_hash, embedding in zip(hash_to_text_map.keys(), embedding_responses):
                embedding_vector = embedding.values
                cached_results[text_hash] = embedding_vector
                rows_to_insert.append({"chunk_hash": text_hash, "embedding": embedding_vector})
            
            if rows_to_insert:
                errors = bq_client.insert_rows_json(cache_table_id, rows_to_insert)
                if errors: print(f"Error inserting into cache: {errors}")

        for text_hash in hashes_to_check:
            embedding_vectors.append(cached_results[text_hash])

    print(f"  -> Embeddings retrieved. Made {api_calls} new API calls.")
    return embedding_vectors

def process_book(book_id, embedding_model, bq_client, cache_table_id):
    """
    Downloads, chunks, embeds, and builds a local FAISS index for a book.
    """
    try:
        # 1. Get Metadata
        print(f"  -> Fetching metadata for book {book_id}...")
        book_metadata = get_book_metadata(book_id)
        if not book_metadata: return

        # 2. Download Text
        book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"  -> Downloading {book_url}")
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        raw_text = response.text

        # 3. Clean and Chunk
        print("  -> Cleaning & chunking text...")
        cleaned_text = clean_gutenberg_text(raw_text)
        chunks = chunk_text_by_paragraph(cleaned_text)
        if not chunks: 
            print(f"  -> No content chunks found for book {book_id}. Skipping.")
            return

        # 4. Get Embeddings
        embedding_vectors = get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, chunks)

        # 5. Build and Save FAISS Index
        db_vectors = np.array(embedding_vectors, dtype=np.float32)
        index = faiss.IndexFlatL2(db_vectors.shape[1])
        index.add(db_vectors)
        index_path = f"app/local_index_{book_id}.faiss"
        faiss.write_index(index, index_path)
        print(f"  -> FAISS index saved to '{index_path}'")

        # 6. Map to Schema and Save Metadata
        all_metadata = [map_gutenberg_to_schema(book_metadata, chunk, i) for i, chunk in enumerate(chunks)]
        metadata_path = f"app/local_metadata_{book_id}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)
        print(f"  -> Metadata saved to '{metadata_path}'")

    except Exception as e:
        print(f"An error occurred while processing book {book_id}: {e}")

def main():
    """Main function to orchestrate the full ingestion process."""
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    cache_table_id = os.getenv("BIGQUERY_CACHE_TABLE")
    
    if not all([project_id, location, cache_table_id]):
        print("Error: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and BIGQUERY_CACHE_TABLE must be set in .env")
        return

    print("Configuration loaded successfully.")
    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    bq_client = bigquery.Client(project=project_id)
    print("Vertex AI and BigQuery clients initialized.")

    manifest_path = os.path.join(os.path.dirname(__file__), 'gutenberg_manifest.txt')
    book_ids = load_book_ids_from_manifest(manifest_path)
    if not book_ids: return

    print(f"\nStarting ingestion for {len(book_ids)} books (processing first one only)...")
    for book_id in book_ids[:1]:
        print(f"Processing book ID: {book_id}")
        process_book(book_id, embedding_model, bq_client, cache_table_id)
        time.sleep(1)

    print("\nGutenberg harvester process complete.")

if __name__ == "__main__":
    main()
