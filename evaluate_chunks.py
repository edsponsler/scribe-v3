import requests
import sys
import os

# Add the 'ingestion' directory to the Python path to import our processor
sys.path.append(os.path.join(os.path.dirname(__file__), 'ingestion'))

from text_processor import clean_gutenberg_text, chunk_text_by_paragraph

def evaluate_book_chunks(book_id: str):
    """
    Downloads, cleans, and chunks a book, then saves the chunks to a local file for evaluation.
    """
    print(f"Starting evaluation for Book ID: {book_id}")

    # 1. Download the book text
    try:
        book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f" -> Downloading from {book_url}")
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        raw_text = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading book {book_id}: {e}")
        return

    # 2. Clean and chunk the text using our existing functions
    print(" -> Cleaning text...")
    cleaned_text = clean_gutenberg_text(raw_text)

    print(" -> Chunking text...")
    chunks = chunk_text_by_paragraph(cleaned_text)

    if not chunks:
        print("No chunks were produced after processing.")
        return

    # 3. Save the chunks to a local file for review
    output_filename = f"{book_id}_chunks_evaluation.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i+1} (Length: {len(chunk)} characters) ---\n")
            f.write(chunk)
            f.write("\n\n")

    print("\n--- EVALUATION COMPLETE ---")
    print(f" -> Found {len(chunks)} chunks.")
    print(f" -> Review the output in the file: {output_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_chunks.py <BOOK_ID>")
        sys.exit(1)
    
    book_to_evaluate = sys.argv[1]
    evaluate_book_chunks(book_to_evaluate)