import os
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

# --- Pericope Map for Genesis ---
PERICOPE_MAP = {
    "Bereshit": "Genesis 1:1-6:8",
    "Noach": "Genesis 6:9-11:32",
    "Lech Lecha": "Genesis 12:1-17:27",
    "Vayera": "Genesis 18:1-22:24",
    "Chayei Sara": "Genesis 23:1-25:18",
    "Toldot": "Genesis 25:19-28:9",
    "Vayetzei": "Genesis 28:10-32:3",
    "Vayishlach": "Genesis 32:4-36:43",
    "Vayeshev": "Genesis 37:1-40:23",
    "Miketz": "Genesis 41:1-44:17",
    "Vayigash": "Genesis 44:18-47:27",
    "Vayechi": "Genesis 47:28-50:26"
}

def map_sefaria_to_schema(sefaria_data, chunk_text, chunk_index, chunk_level, parent_id=None):
    """Maps Sefaria API response to the SCRIBE metadata schema."""
    ref = sefaria_data.get('ref')
    mapped_data = {
        "id": str(uuid.uuid4()),
        "source_id": f"sefaria:{ref}",
        "title": sefaria_data.get('book'),
        "canonical_reference": f"{ref}:{chunk_index + 1}" if chunk_level == "verse" else ref,
        "text_type": sefaria_data.get('type'),
        "chunk_level": chunk_level,
        "parent_chunk_id": parent_id,
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

def flatten_text_array(text_array):
    """Flattens a nested list of strings into a single list of strings."""
    if isinstance(text_array, str):
        return [text_array]
    flat_list = []
    for item in text_array:
        if isinstance(item, str):
            flat_list.append(item)
        elif isinstance(item, list):
            flat_list.extend(flatten_text_array(item))
    return flat_list

def process_pericope(pericope_name, pericope_ref, embedding_model, bq_client, cache_table_id):
    """
    Fetches, chunks, embeds, and builds a local FAISS index for a Sefaria pericope.
    """
    try:
        # 1. Fetch Full Pericope Data
        url = f"https://www.sefaria.org/api/texts/{pericope_ref}"
        print(f"  -> Fetching data for {pericope_name} ({pericope_ref})")
        response = requests.get(url)
        response.raise_for_status()
        sefaria_data = response.json()

        # 2. Create Parent Chunk (Pericope)
        raw_text = sefaria_data.get('text', [])
        child_chunks = flatten_text_array(raw_text)
        parent_text = " ".join(child_chunks)
        parent_metadata = map_sefaria_to_schema(sefaria_data, parent_text, 0, "pericope")
        parent_id = parent_metadata['id']

        # 3. Create Child Chunks (Verses)
        child_metadata_list = []
        for i, child_chunk in enumerate(child_chunks):
            child_metadata = map_sefaria_to_schema(sefaria_data, child_chunk, i, "verse", parent_id)
            child_metadata_list.append(child_metadata)

        # 4. Get Embeddings for Verses
        verse_texts = [m['text'] for m in child_metadata_list]
        verse_embeddings = get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, verse_texts)

        return parent_metadata, child_metadata_list, verse_embeddings

    except Exception as e:
        print(f"An error occurred while processing {pericope_name}: {e}")
        return None, None, None

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

    # --- Clean up old index files ---
    old_index_files = [
        "app/local_index_genesis.faiss",
        "app/local_metadata_genesis.pkl"
    ]
    for f in old_index_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed old index file: {f}")

    all_verse_metadata = []
    all_verse_embeddings = []
    pericope_metadata_store = {}

    print(f"\nStarting hierarchical ingestion for the book of Genesis...")
    for pericope_name, pericope_ref in PERICOPE_MAP.items():
        parent_metadata, child_metadata_list, verse_embeddings = process_pericope(pericope_name, pericope_ref, embedding_model, bq_client, cache_table_id)

        if parent_metadata and child_metadata_list and verse_embeddings:
            # Store pericope metadata
            pericope_metadata_store[parent_metadata['id']] = parent_metadata
            # Collect verse data
            all_verse_metadata.extend(child_metadata_list)
            all_verse_embeddings.extend(verse_embeddings)

    # --- Build and Save Verse Index ---
    if all_verse_embeddings:
        db_vectors = np.array(all_verse_embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(db_vectors.shape[1])
        index.add(db_vectors)

        verse_index_path = "app/local_verses_genesis.faiss"
        faiss.write_index(index, verse_index_path)
        print(f"  -> Verse FAISS index for Genesis created and saved to '{verse_index_path}'")

        # --- Save Verse Metadata ---
        verse_metadata_path = "app/local_verses_genesis.pkl"
        with open(verse_metadata_path, 'wb') as f:
            pickle.dump(all_verse_metadata, f)
        print(f"  -> Verse metadata for Genesis saved to '{verse_metadata_path}'")


    # --- Save Pericope Metadata Store ---
    if pericope_metadata_store:
        pericope_store_path = "app/local_pericopes_genesis.pkl"
        with open(pericope_store_path, 'wb') as f:
            pickle.dump(pericope_metadata_store, f)
        print(f"  -> Pericope metadata store for Genesis saved to '{pericope_store_path}'")


    print("\nPericope harvester process complete.")

if __name__ == "__main__":
    main()
