import sys
import requests
import uuid

sys.path.append('ingestion')
from ingestion.sefaria_harvester import get_text_for_reference

def test_sefaria_chapter_chunking(book_name, chapter_to_inspect):
    """Tests the chapter-based parent/child chunking logic for Sefaria texts."""
    print(f"--- Testing Sefaria Chapter Chunking for: {book_name} {chapter_to_inspect} ---")

    chapter_ref = f"{book_name} {chapter_to_inspect}"
    chapter_data = get_text_for_reference(chapter_ref)
    if not chapter_data or not chapter_data.get('text'):
        print(f"Error: Could not retrieve text for {chapter_ref}.")
        return

    parent_id = str(uuid.uuid4()) # Dummy parent ID for the chapter
    verses = chapter_data.get('text', [])
    
    print(f"-> Found {len(verses)} verses in this chapter.")
    print("--- Metadata for first 5 verses ---")
    
    for verse_num, verse_text in enumerate(verses[:5]):
        verse_ref = f"{book_name} {chapter_to_inspect}:{verse_num + 1}"
        child_metadata = {
            "chunk_id": str(uuid.uuid4()),
            "parent_chunk_id": parent_id,
            "canonical_reference": verse_ref,
            "text": verse_text[:60] + "..."
        }
        print(f"  Verse {verse_num + 1}: {child_metadata}")
    print("-------------------------------------")

if __name__ == "__main__":
    test_sefaria_chapter_chunking("Genesis", 50)
