import os
import requests
import json
import uuid
import hashlib
import pickle
import numpy as np
import faiss
import argparse
from dotenv import load_dotenv

# --- Vertex AI & BigQuery Imports ---
import vertexai
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from ingestion.text_processor import chunk_text_by_paragraph # Assuming Sefaria text can be chunked this way

def map_sefaria_to_schema(sefaria_data, chunk_text, chunk_index):
    """Maps Sefaria API response to the SCRIBE metadata schema."""
    ref = sefaria_data.get('ref')
    mapped_data = {
        "id": str(uuid.uuid4()),
        "source_id": f"sefaria:{ref}",
        "title": sefaria_data.get('book'),
        "author": None, # Not typically available at this level
        "publisher": sefaria_data.get('publisher'),
        "source_url": f"https://www.sefaria.org/{ref.replace(' ', '_')}",
        "language": "en", # Assuming English text for now
        "canonical_reference": f"{ref}:{chunk_index + 1}",
        "text_type": sefaria_data.get('type'),
        "chunk_level": "verse", # Or could be 'paragraph' depending on chunking
        "text": chunk_text
    }
    return {k: v for k, v in mapped_data.items() if v is not None}

def get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, chunks):
    """Gets embeddings for a list of text chunks, using a BigQuery cache."""
    print(f"  -> Getting embeddings for {len(chunks)} chunks, using BigQuery cache...")
    embedding_vectors = []
    api_calls = 0
    BATCH_SIZE = 50

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

def process_sefaria_text(reference, embedding_model, bq_client, cache_table_id):
    """
    Fetches, chunks, embeds, and builds a local FAISS index for a Sefaria text.
    """
    try:
        # 1. Fetch Data
        url = f"https://www.sefaria.org/api/texts/{reference}"
        print(f"  -> Fetching data from {url}")
        response = requests.get(url)
        response.raise_for_status()
        sefaria_data = response.json()

        # 2. Extract and Chunk Text
        # The 'text' field can be a list of strings (verses/paragraphs)
        raw_text_chunks = sefaria_data.get('text', [])
        if not isinstance(raw_text_chunks, list):
            raw_text_chunks = [str(raw_text_chunks)] # Ensure it's a list
        
        # Simple cleaning (can be improved)
        chunks = [text.strip() for text in raw_text_chunks if text.strip()]
        if not chunks:
            print(f"  -> No content chunks found for {reference}. Skipping.")
            return

        # 3. Get Embeddings
        embedding_vectors = get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, chunks)

        # 4. Build and Save FAISS Index
        db_vectors = np.array(embedding_vectors, dtype=np.float32)
        index = faiss.IndexFlatL2(db_vectors.shape[1])
        index.add(db_vectors)
        index_path = f"app/local_index_sefaria_{reference.replace(' ', '_')}.faiss"
        faiss.write_index(index, index_path)
        print(f"  -> FAISS index saved to '{index_path}'")

        # 5. Map to Schema and Save Metadata
        all_metadata = [map_sefaria_to_schema(sefaria_data, chunk, i) for i, chunk in enumerate(chunks)]
        metadata_path = f"app/local_metadata_sefaria_{reference.replace(' ', '_')}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)
        print(f"  -> Metadata saved to '{metadata_path}'")

    except Exception as e:
        print(f"An error occurred while processing {reference}: {e}")

def main():
    """Main function to orchestrate the full ingestion process."""
    parser = argparse.ArgumentParser(description="Fetch and process text from the Sefaria API.")
    parser.add_argument("reference", type=str, help="The Sefaria reference to fetch (e.g., 'Genesis 1:1').")
    args = parser.parse_args()

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

    print(f"\nProcessing Sefaria reference: {args.reference}")
    process_sefaria_text(args.reference, embedding_model, bq_client, cache_table_id)

    print("\nSefaria harvester process complete.")

if __name__ == "__main__":
    main()