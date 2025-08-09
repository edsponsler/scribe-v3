import sys
import requests
import json

def inspect_sefaria_response(book_name):
    """Fetches and prints the entire JSON response for a Sefaria book."""
    print(f"--- Inspecting Sefaria Response for: {book_name} ---")
    try:
        url = f"https://www.sefaria.org/api/texts/{book_name}"
        response = requests.get(url)
        response.raise_for_status()
        book_data = response.json()
        
        print(json.dumps(book_data, indent=2))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_sefaria_response("Genesis")