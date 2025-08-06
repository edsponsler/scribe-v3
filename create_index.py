import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from utils import update_env_file

# Load environment variables from your .env file
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Initialize the Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# The GCS path where the index files will be stored
# We use a new folder to keep it clean
index_contents_uri = f"gs://{GCS_BUCKET_NAME}/vector_search_index"

print(f"Creating Brute Force index in project {PROJECT_ID}...")

# This is a dedicated helper method to create a simple index.
# It's much simpler than the generic 'create' method.
scribe_index = aiplatform.MatchingEngineIndex.create_brute_force_index(
    display_name="scribe-v3-index",
    contents_delta_uri=index_contents_uri,
    dimensions=768,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="STREAM_UPDATE"
)

print(f"Index created successfully: {scribe_index.resource_name}")
update_env_file("VERTEX_INDEX_ID", scribe_index.name)