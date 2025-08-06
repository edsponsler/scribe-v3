import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from utils import update_env_file

# Load environment variables from your .env file
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# Initialize the Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)

print(f"Creating Index Endpoint in project {PROJECT_ID}...")

index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="scribe-v3-index-endpoint",
    public_endpoint_enabled=True,
)

print(f"Index Endpoint created successfully: {index_endpoint.resource_name}")
update_env_file("VERTEX_ENDPOINT_ID", index_endpoint.name)