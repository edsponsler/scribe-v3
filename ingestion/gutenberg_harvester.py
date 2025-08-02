import os
import re
import time # <--- Add this new import
import requests
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.exceptions import NotFound

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

# ---- START: NEW FUNCTION ----
def process_book(book_id, bucket):
    """
    Downloads a book's plain text from Gutenberg and uploads it to GCS.

    Args:
        book_id (str): The Project Gutenberg ID of the book.
        bucket (storage.Bucket): The GCS bucket object to upload to.
    """
    try:
        # Use the more modern /cache/epub/ URL format
        book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"  -> Downloading {book_url}")
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        
        # The content is UTF-8 encoded plain text
        book_text = response.text

        # Create a new blob (file) in GCS
        blob = bucket.blob(f"{book_id}.txt")
        
        # Upload the text
        blob.upload_from_string(book_text, content_type="text/plain")
        
        print(f"  -> Successfully uploaded {book_id}.txt to GCS.")

    except requests.exceptions.HTTPError as e:
        print(f"  -> HTTP Error for book {book_id}: {e}. It may not be available as plain text.")
    except requests.exceptions.RequestException as e:
        print(f"  -> Network Error downloading book {book_id}: {e}")
# ---- END: NEW FUNCTION ----

def main():
    """Main function to orchestrate the download and upload process."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not project_id or not bucket_name:
        print("Error: GOOGLE_CLOUD_PROJECT and GCS_BUCKET_NAME must be set.")
        return

    print("Configuration loaded successfully.")

    # ---- START: MODIFIED CODE ----
    # Set up GCS client and bucket
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.get_bucket(bucket_name)
    except NotFound:
        print(f"Error: The bucket '{bucket_name}' does not exist.")
        return
        
    manifest_path = os.path.join(os.path.dirname(__file__), 'gutenberg_manifest.txt')
    book_ids = load_book_ids_from_manifest(manifest_path)

    if not book_ids:
        print("Could not load book IDs from manifest. Exiting.")
        return

    print(f"\nStarting ingestion for {len(book_ids)} books...")
    
    for book_id in book_ids:
        print(f"Processing book ID: {book_id}")
        process_book(book_id, bucket)
        # Be polite to the server
        time.sleep(1)

    print("\nIngestion process complete.")
    # ---- END: MODIFIED CODE ----


if __name__ == "__main__":
    main()