import os
import sys
import time
import requests
import hashlib
from dotenv import load_dotenv

# --- Cloud Imports ---
import vertexai
from google.cloud import aiplatform
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# Add the 'ingestion' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ingestion'))
from text_processor import clean_gutenberg_text, chunk_text_by_paragraph


def reindex_book(book_id: str):
    """
    Selectively re-indexes a single book using an embedding cache and batch processing.
    """
    print(f"--- Starting CACHE-OPTIMIZED re-indexing for Book ID: {book_id} ---")

    # --- Load Config and Initialize Clients ---
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    index_id = os.getenv("VERTEX_INDEX_ID")

    if not all([project_id, location, index_id]):
        print("Error: Required environment variables are not set.")
        return

    vertexai.init(project=project_id, location=location)
    scribe_index = aiplatform.MatchingEngineIndex(index_name=index_id)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    bq_client = bigquery.Client()
    cache_table_id = f"{project_id}.scribe_v3_dataset.embedding_cache"
    print("Clients initialized.")
    
    # --- 1. Generate New Chunks ---
    print("-> Generating new chunks based on current strategy...")
    try:
        book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        cleaned_text = clean_gutenberg_text(response.text)
        new_chunks = chunk_text_by_paragraph(cleaned_text)
        print(f"-> Generated {len(new_chunks)} new chunks.")
    except Exception as e:
        print(f"Error fetching or processing book: {e}")
        return

    # --- 2. Get Embeddings (Using Cache) ---
    print("-> Getting embeddings for new chunks, using cache...")
    # Initialize the list to hold our final datapoints
    new_datapoints = []
    api_calls = 0
    BATCH_SIZE = 100 # Batch size for API calls

    # Process chunks in batches to be efficient with API calls
    for i in range(0, len(new_chunks), BATCH_SIZE):
        chunk_batch = new_chunks[i:i + BATCH_SIZE]
        
        # Check cache for the entire batch first
        hashes_to_check = [hashlib.sha256(text.encode()).hexdigest() for text in chunk_batch]
        query = f"SELECT chunk_hash, embedding FROM `{cache_table_id}` WHERE chunk_hash IN UNNEST(@hashes)"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hashes", "STRING", hashes_to_check)]
        )
        cached_results = {row.chunk_hash: row.embedding for row in bq_client.query(query, job_config=job_config)}
        
        # Determine which chunks in the batch are new
        new_texts_for_batch = []
        hash_to_text_map = {}
        for text, text_hash in zip(chunk_batch, hashes_to_check):
            if text_hash not in cached_results:
                new_texts_for_batch.append(text)
                hash_to_text_map[text_hash] = text
        
        # If there are new chunks, get their embeddings from the API
        if new_texts_for_batch:
            api_calls += len(new_texts_for_batch)
            embedding_responses = embedding_model.get_embeddings(new_texts_for_batch)
            
            rows_to_insert = []
            for text_hash, embedding in zip(hash_to_text_map.keys(), embedding_responses):
                embedding_vector = embedding.values
                cached_results[text_hash] = embedding_vector
                rows_to_insert.append({"chunk_hash": text_hash, "embedding": embedding_vector})
            
            # Save the newly generated embeddings to the cache
            errors = bq_client.insert_rows_json(cache_table_id, rows_to_insert)
            if errors: print(f"Error inserting into cache: {errors}")

        # Assemble the full list of datapoints for this batch
        for text, text_hash in zip(chunk_batch, hashes_to_check):
             # The index 'i' gives the overall chunk number for the book
            datapoint_index = i + hashes_to_check.index(text_hash)
            new_datapoints.append({
                "datapoint_id": f"{book_id}-{datapoint_index}",
                "feature_vector": cached_results[text_hash]
            })

    print(f"-> Embeddings retrieved. Made {api_calls} new API calls.")

    # --- 3. Update Vector Search Index ---
    if new_datapoints:
        print(f"-> Deleting old datapoints for book {book_id} from Vector Search...")
        ids_to_delete = [f"{book_id}-{i}" for i in range(50000)]
        
        # Batch deletion
        for i in range(0, len(ids_to_delete), 1000):
            batch = ids_to_delete[i:i + 1000]
            scribe_index.remove_datapoints(datapoint_ids=batch)
        print("-> Deletion requests sent. Waiting for propagation...")
        time.sleep(180)

        print(f"-> Upserting {len(new_datapoints)} new datapoints...")
        # Batch upsertion
        for i in range(0, len(new_datapoints), 100):
            batch = new_datapoints[i:i + 100]
            scribe_index.upsert_datapoints(datapoints=batch)
            print(f"  -> Upserted batch of {len(batch)} datapoints.")

    print(f"--- Re-indexing for Book ID: {book_id} complete ---")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reindex_book.py <BOOK_ID>")
        sys.exit(1)
    
    book_to_reindex = sys.argv[1]
    reindex_book(book_to_reindex)