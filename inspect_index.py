import pickle
import os

METADATA_PATH = os.path.join(os.path.dirname(__file__), 'app', 'local_metadata_genesis.pkl')

def inspect_genesis_metadata():
    """
    Loads the Genesis metadata and prints a sample of the hierarchical structure.
    """
    print(f"--- Inspecting Metadata from {METADATA_PATH} ---")

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    with open(METADATA_PATH, 'rb') as f:
        all_metadata = pickle.load(f)

    # Find the first pericope (parent chunk)
    parent_chunk = None
    for metadata in all_metadata:
        if metadata.get('chunk_level') == 'pericope':
            parent_chunk = metadata
            break

    if not parent_chunk:
        print("Error: No pericope (parent) chunks found in the metadata.")
        return

    print("\n--- Found Parent Chunk (Pericope) ---")
    print(f"  ID: {parent_chunk['id']}")
    print(f"  Reference: {parent_chunk['canonical_reference']}")
    print(f"  Chunk Level: {parent_chunk['chunk_level']}")
    print(f"  Text Preview: {parent_chunk['text'][:100]}...")

    # Find its children
    parent_id = parent_chunk['id']
    child_chunks = []
    for metadata in all_metadata:
        if metadata.get('parent_chunk_id') == parent_id:
            child_chunks.append(metadata)

    print(f"\n--- Found {len(child_chunks)} Child Chunks (Verses) ---")
    print("(Showing first 3)")

    for i, child in enumerate(child_chunks[:3]):
        print(f"\n  -> Child {i+1}")
        print(f"    ID: {child['id']}")
        print(f"    Parent ID: {child['parent_chunk_id']}")
        print(f"    Reference: {child['canonical_reference']}")
        print(f"    Chunk Level: {child['chunk_level']}")
        print(f"    Text: {child['text']}")

    print("\n--- Inspection Complete ---")

if __name__ == "__main__":
    inspect_genesis_metadata()
