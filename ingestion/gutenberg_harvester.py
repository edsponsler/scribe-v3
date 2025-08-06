import os
import time
import requests
from dotenv import load_dotenv

# --- Vertex AI Imports ---
import vertexai
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from ingestion.text_processor import clean_gutenberg_text, chunk_text_by_paragraph

def load_book_ids_from_manifest(manifest_path):
    """Loads book IDs from a text manifest file."""
    try:
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
            book_ids = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    book_ids.append(line)
            return book_ids
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return []

def process_book(book_id, embedding_model, scribe_index):
    """
    Downloads, chunks, embeds, and upserts a book's content to the index.
    """
    try:
        book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"  -> Downloading {book_url}")
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        raw_text = response.text

        print("  -> Cleaning & chunking text...")
        cleaned_text = clean_gutenberg_text(raw_text)
        chunks = chunk_text_by_paragraph(cleaned_text)

        if not chunks:
            print(f"  -> No content chunks found for book {book_id}. Skipping.")
            return

        print(f"  -> Found {len(chunks)} chunks. Embedding and upserting the first 5...")

        target_chunks = chunks[:5]
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in target_chunks]
        embeddings = embedding_model.get_embeddings(inputs)

        datapoints = []
        for i, embedding in enumerate(embeddings):
            datapoints.append(
                {
                    "datapoint_id": f"{book_id}-{i}",
                    "feature_vector": embedding.values, # The correct key is "feature_vector"
                }
            )
        
        scribe_index.upsert_datapoints(datapoints=datapoints)
        print(f"  -> Successfully upserted {len(datapoints)} datapoints.")

    except Exception as e:
        print(f"An error occurred while processing book {book_id}: {e}")


def main():
    """Main function to orchestrate the full ingestion process."""
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    index_id = os.getenv("VERTEX_INDEX_ID")
    
    if not all([project_id, location, index_id]):
        print("Error: Required environment variables are not set.")
        return

    print("Configuration loaded successfully.")

    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    scribe_index = aiplatform.MatchingEngineIndex(index_name=index_id)
    print("Vertex AI clients initialized.")

    manifest_path = os.path.join(os.path.dirname(__file__), 'gutenberg_manifest.txt')
    book_ids = load_book_ids_from_manifest(manifest_path)
    if not book_ids: return

    print(f"\nStarting ingestion for {len(book_ids)} books...")

    for book_id in book_ids[:1]:
        print(f"Processing book ID: {book_id}")
        process_book(book_id, embedding_model, scribe_index)
        time.sleep(1)

    print("\nObjective 3 process complete.")


if __name__ == "__main__":
    main()