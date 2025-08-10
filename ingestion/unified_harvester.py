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
import re

# --- Vertex AI & BigQuery Imports ---
import vertexai
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel as TextGenerationModel

# --- Constants ---
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
BOOK_NAME = "Genesis"
CHAPTER_COUNT = 50

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

def generate_summary(text_generation_model, text_to_summarize, entity_name):
    """Generates a ~100 word summary for a given text."""
    print(f"    -> Generating summary for {entity_name}...")
    try:
        prompt = f"""Summarize the following biblical passage from the book of Genesis in approximately 100 words.
        Focus on the key events, characters, and theological points.

        Passage:
        {text_to_summarize}

        Summary:
        """
        response = text_generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"      -> Error generating summary for {entity_name}: {e}")
        return None

def parse_ref(ref_string):
    """Parses a Sefaria reference string to get book, chapter, and verse."""
    match = re.match(r"([a-zA-Z ]+) (\d+):(\d+)", ref_string)
    if match:
        book, chapter, verse = match.groups()
        return book.strip(), int(chapter), int(verse)
    return None, None, None

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

# --- Main Processing Logic ---

def main():
    """Main function to orchestrate the full unified ingestion process."""
    # --- Setup ---
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    cache_table_id = os.getenv("BIGQUERY_CACHE_TABLE")
    generative_model_name = os.getenv("VERTEX_GENERATIVE_MODEL", "gemini-1.5-flash")

    if not all([project_id, location, cache_table_id]):
        print("Error: Required environment variables must be set.")
        return

    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    text_generation_model = TextGenerationModel(generative_model_name)
    bq_client = bigquery.Client(project=project_id)
    print(f"--- Using generative model: {generative_model_name} for summaries ---")

    # --- Data Stores Initialization with Caching ---
    app_dir = "app"
    chapters_path = os.path.join(app_dir, "local_chapters_genesis.pkl")
    pericopes_path = os.path.join(app_dir, "local_pericopes_genesis.pkl")

    chapters_store = pickle.load(open(chapters_path, "rb")) if os.path.exists(chapters_path) else {}
    pericopes_store = pickle.load(open(pericopes_path, "rb")) if os.path.exists(pericopes_path) else {}
    verses_store = {} 

    # --- Phase 1: Process Chapters and build Verse Store ---
    print("\n--- Phase 1: Processing all chapters and verses ---")
    for chap_num in range(1, CHAPTER_COUNT + 1):
        chapter_ref = f"{BOOK_NAME} {chap_num}"
        print(f"  -> Processing {chapter_ref}")
        
        if chapter_ref not in chapters_store:
            api_data = get_text_for_reference(chapter_ref)
            if not api_data or not api_data.get('text'): continue
            
            chapters_store[chapter_ref] = {
                "id": str(uuid.uuid4()),
                "source_id": "sefaria:genesis",
                "title": BOOK_NAME,
                "language": "en",
                "canonical_reference": chapter_ref,
                "chunk_level": "chapter",
                "text": " ".join(api_data['text']),
                "summary": None
            }

        chapter_id = chapters_store[chapter_ref]["id"]
        api_data = get_text_for_reference(chapter_ref) # Re-fetch for verse processing
        if api_data and api_data.get('text'):
            for verse_num, verse_text in enumerate(api_data.get('text', [])):
                verse_ref = f"{chapter_ref}:{verse_num + 1}"
                if verse_ref not in verses_store:
                    verses_store[verse_ref] = {
                        "id": str(uuid.uuid4()),
                        "source_id": "sefaria:genesis",
                        "title": BOOK_NAME,
                        "language": "en",
                        "canonical_reference": verse_ref,
                        "chunk_level": "verse",
                        "text": verse_text,
                        "parent_chunk_ids": []
                    }
                if chapter_id not in verses_store[verse_ref]["parent_chunk_ids"]:
                    verses_store[verse_ref]["parent_chunk_ids"].append(chapter_id)

    # --- Phase 2: Process Pericopes and Link Verses ---
    print("\n--- Phase 2: Processing all pericopes and linking verses ---")
    for pericope_name, pericope_ref in PERICOPE_MAP.items():
        print(f"  -> Processing {pericope_name} ({pericope_ref})")
        
        if pericope_name not in pericopes_store:
            api_data = get_text_for_reference(pericope_ref)
            if not api_data or not api_data.get('text'): continue

            pericopes_store[pericope_name] = {
                "id": str(uuid.uuid4()),
                "source_id": "sefaria:genesis",
                "title": BOOK_NAME,
                "language": "en",
                "canonical_reference": pericope_ref,
                "chunk_level": "pericope",
                "text": " ".join(flatten_text_array(api_data.get('text', []))),
                "summary": None
            }

        pericope_id = pericopes_store[pericope_name]["id"]
        start_chap = int(re.search(r'(\d+):', pericope_ref).group(1))
        end_chap = int(re.search(r'-(\d+):', pericope_ref).group(1))

        for verse_ref, verse_data in verses_store.items():
            book, chapter, verse = parse_ref(verse_ref)
            if start_chap <= chapter <= end_chap:
                 if pericope_id not in verse_data["parent_chunk_ids"]:
                    verse_data["parent_chunk_ids"].append(pericope_id)

    # --- Phase 3: Generate Summaries (with Caching) ---
    print("\n--- Phase 3: Generating summaries for chapters and pericopes ---")
    summary_metadata_list = []
    for store in [chapters_store, pericopes_store]:
        for key, data in store.items():
            if not data.get("summary"):
                summary = generate_summary(text_generation_model, data["text"], data["canonical_reference"])
                data["summary"] = summary
            if data.get("summary"):
                summary_metadata_list.append({
                    "id": data["id"],
                    "canonical_reference": data["canonical_reference"],
                    "chunk_level": data.get("chunk_level") or data.get("type"), # Handle old key
                    "text": data["summary"]
                })

    # --- Phase 4: Generate Embeddings and Save Artifacts ---
    print("\n--- Phase 4: Generating embeddings and saving all artifacts ---")
    
    # 4a. Unified Summaries Index
    summary_texts = [s['text'] for s in summary_metadata_list]
    summary_embeddings = get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, summary_texts)
    
    if summary_embeddings:
        summary_vectors = np.array(summary_embeddings, dtype=np.float32)
        summary_index = faiss.IndexFlatL2(summary_vectors.shape[1])
        summary_index.add(summary_vectors)
        faiss.write_index(summary_index, os.path.join(app_dir, "local_summaries_genesis.faiss"))
        with open(os.path.join(app_dir, "local_summaries_genesis.pkl"), "wb") as f:
            pickle.dump(summary_metadata_list, f)
        print("  -> Unified summaries index and metadata saved.")

    # 4b. Verses Index
    verse_list = list(verses_store.values())
    verse_texts = [v['text'] for v in verse_list]
    verse_embeddings = get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, verse_texts)

    if verse_embeddings:
        verse_vectors = np.array(verse_embeddings, dtype=np.float32)
        verse_index = faiss.IndexFlatL2(verse_vectors.shape[1])
        verse_index.add(verse_vectors)
        faiss.write_index(verse_index, os.path.join(app_dir, "local_verses_genesis.faiss"))
        with open(os.path.join(app_dir, "local_verses_genesis.pkl"), "wb") as f:
            pickle.dump(verse_list, f)
        print("  -> Verse index and metadata saved.")

    # 4c. Parent Retrieval Stores
    with open(chapters_path, "wb") as f:
        pickle.dump(chapters_store, f)
    print("  -> Chapters retrieval store saved.")
    
    with open(pericopes_path, "wb") as f:
        pickle.dump(pericopes_store, f)
    print("  -> Pericopes retrieval store saved.")

    print("\n--- Unified harvester process complete. ---")


if __name__ == "__main__":
    main()
