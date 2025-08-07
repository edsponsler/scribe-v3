import os
import argparse
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from prompts import CORE_PERSONA_PROMPT, DECOMPOSITION_PROMPT_TEMPLATE, SYNTHESIS_PROMPT_TEMPLATE

def decompose_query(topic, model_name):
    """Uses a Gemini model to decompose a complex topic into sub-questions."""
    print(f"--- Decomposing Topic: '{topic}' ---")

    prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(
        core_persona=CORE_PERSONA_PROMPT,
        topic=topic
    )

    try:
        model = GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        print("\n--- Generated Research Plan ---")
        print(response.text)
        return response.text.strip().split('\n')

    except Exception as e:
        print(f"An error occurred during query decomposition: {e}")
        return []

import faiss
import numpy as np
import pickle

from vertexai.language_models import TextEmbeddingModel

INDEX_PATH = os.path.join(os.path.dirname(__file__), 'app', 'local_index_genesis.faiss')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'app', 'local_metadata_genesis.pkl')

def iterative_retrieval(sub_questions, embedding_model, index, metadata):
    """Performs vector search for each sub-question to gather context."""
    print("\n--- Starting Iterative Retrieval ---")
    
    all_retrieved_chunks = []
    for question in sub_questions:
        if not question.strip(): continue
        print(f"\n  -> Searching for: '{question.strip()}'")

        try:
            # Embed the sub-question
            question_embedding = embedding_model.get_embeddings([question])[0].values
            
            # Perform FAISS search
            k = 5 # Number of results to retrieve
            distances, indices = index.search(np.array([question_embedding], dtype=np.float32), k)

            print(f"    -> Found {len(indices[0])} potential matches.")
            retrieved_for_question = []
            retrieved_for_question = []
            for j, i in enumerate(indices[0]):
                if i < len(metadata):
                    chunk_metadata = metadata[i]
                    retrieved_for_question.append(chunk_metadata)
                    print(f"      - Retrieved: {chunk_metadata.get('canonical_reference')} (Score: {distances[0][j]:.4f})")
                else:
                    print(f"      - Warning: FAISS index {i} is out of bounds for metadata list.")
            
            all_retrieved_chunks.extend(retrieved_for_question)

        except Exception as e:
            print(f"    -> An error occurred during retrieval: {e}")

    return all_retrieved_chunks

def final_synthesis(topic, retrieved_chunks, model_name):
    """Synthesizes the retrieved context into a final essay."""
    print("\n--- Starting Final Synthesis ---")

    # Construct the context string
    context_string = ""
    for chunk in retrieved_chunks:
        context_string += f"Reference: {chunk.get('canonical_reference')}\n"
        context_string += f"Text: {chunk.get('text')}\n\n"

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        core_persona=CORE_PERSONA_PROMPT,
        topic=topic,
        context=context_string
    )

    try:
        model = GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        print("\n--- Generated Research Paper ---")
        print(response.text)

    except Exception as e:
        print(f"An error occurred during final synthesis: {e}")


def main():
    """Main function to run the deep research process."""
    parser = argparse.ArgumentParser(description="Perform a deep research query on a topic.")
    parser.add_argument("topic", type=str, help="The complex research topic to investigate.")
    args = parser.parse_args()

    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    model_name = os.getenv("VERTEX_GENERATIVE_MODEL")

    if not all([project_id, location, model_name]):
        print("Error: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and VERTEX_GENERATIVE_MODEL must be set in .env")
        return

    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    # 1. Decompose Query
    sub_questions = decompose_query(args.topic, model_name)
    if not sub_questions: return

    # 2. Load Index and Metadata
    try:
        print("\n--- Loading Local Index and Metadata ---")
        index = faiss.read_index(INDEX_PATH)
        print(f"  -> Index loaded. Size: {index.ntotal}")
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print(f"  -> Metadata loaded. Size: {len(metadata)}")
    except Exception as e:
        print(f"Error loading local index: {e}")
        return

    # 3. Iterative Retrieval
    retrieved_chunks = iterative_retrieval(sub_questions, embedding_model, index, metadata)

    # 4. Final Synthesis
    final_synthesis(args.topic, retrieved_chunks, model_name)


if __name__ == "__main__":
    main()
