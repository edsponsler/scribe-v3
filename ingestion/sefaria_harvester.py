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

# --- Helper Functions ---

def get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, chunks):
    """Gets embeddings for a list of text chunks, using a BigQuery cache."""
    if not chunks:
        return []
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

# --- Sefaria-Specific Processing ---

def get_text_for_reference(reference):
    """Fetches the text for a specific Sefaria reference."""
    try:
        url = f"https://www.sefaria.org/api/texts/{reference}?context=0"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching text for {reference}: {e}")
        return None

def process_book_from_sefaria(book_name, embedding_model, bq_client, cache_table_id):
    """
    Processes a full book from Sefaria using a chapter-based parent/child chunking strategy.
    """
    print(f"--- Processing book: {book_name} from Sefaria ---")

    retrieval_store = {}
    search_metadata_list = []
    child_texts_for_embedding = []
    total_verses = 0

    # Genesis has 50 chapters
    for chap_num in range(1, 51):
        chapter_ref = f"{book_name} {chap_num}"
        print(f"  -> Processing chapter: {chapter_ref}")
        chapter_data = get_text_for_reference(chapter_ref)
        if not chapter_data or not chapter_data.get('text'):
            print(f"    -> No text found for {chapter_ref}. Skipping.")
            continue

        parent_id = str(uuid.uuid4())
        parent_text = " ".join(chapter_data['text'])

        retrieval_store[parent_id] = {
            "id": parent_id,
            "source_id": f"sefaria:{chapter_ref}",
            "title": book_name,
            "canonical_reference": chapter_ref,
            "text": parent_text,
            "child_chunks": []
        }

        for verse_num, verse_text in enumerate(chapter_data.get('text', [])):
            verse_ref = f"{book_name} {chap_num}:{verse_num + 1}"
            child_id = str(uuid.uuid4())
            total_verses += 1
            
            child_metadata = {
                "chunk_id": child_id,
                "parent_chunk_id": parent_id,
                "canonical_reference": verse_ref,
                "text": verse_text
            }
            search_metadata_list.append(child_metadata)
            retrieval_store[parent_id]['child_chunks'].append(child_metadata)
            child_texts_for_embedding.append(verse_text)

    print(f"\n  -> Total verses processed: {total_verses}")
    embedding_vectors = get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, child_texts_for_embedding)

    # Using "chapters" as the parent level name for clarity
    retrieval_path = f"app/local_chapters_{book_name.lower()}.pkl"
    with open(retrieval_path, 'wb') as f:
        pickle.dump(retrieval_store, f)
    print(f"  -> Retrieval store saved to '{retrieval_path}'")

    db_vectors = np.array(embedding_vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(db_vectors.shape[1])
    index.add(db_vectors)
    index_path = f"app/local_verses_{book_name.lower()}.faiss"
    faiss.write_index(index, index_path)
    print(f"  -> Search index saved to '{index_path}'")

    search_meta_path = f"app/local_verses_{book_name.lower()}.pkl"
    with open(search_meta_path, 'wb') as f:
        pickle.dump(search_metadata_list, f)
    print(f"  -> Search metadata saved to '{search_meta_path}'")

def main():
    parser = argparse.ArgumentParser(description="Fetch and process a full book from the Sefaria API.")
    parser.add_argument("book", type=str, help="The book to process (e.g., 'Genesis').")
    args = parser.parse_args()

    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    cache_table_id = os.getenv("BIGQUERY_CACHE_TABLE")

    if not all([project_id, location, cache_table_id]):
        print("Error: Required environment variables must be set.")
        return

    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    bq_client = bigquery.Client(project=project_id)

    process_book_from_sefaria(args.book, embedding_model, bq_client, cache_table_id)

    print("\nSefaria harvester process complete.")

if __name__ == "__main__":
    main()