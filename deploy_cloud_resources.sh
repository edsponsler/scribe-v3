#!/bin/bash

# This script automates the creation and deployment of Vertex AI Vector Search resources.
# 1. Creates an Index Endpoint.
# 2. Creates a Vector Search Index.
# 3. Deploys the Index to the Endpoint.
# It relies on variables set in the .env file and sourced by startup.sh.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Sourcing environment variables from startup.sh ---"
# Source startup.sh to set GOOGLE_CLOUD_PROJECT, etc.
# and to load any existing variables from .env
source ./startup.sh

echo "--- CREATING INDEX ENDPOINT ---"
python create_endpoint.py

echo "--- CREATING NEW INDEX ---"
python create_index.py

echo "--- Re-sourcing environment variables to get new IDs ---"
# The python scripts have updated .env, so we source again
# to get the new VERTEX_INDEX_ID and VERTEX_ENDPOINT_ID
source ./startup.sh

# Define a unique ID for the deployed index and save it to the .env file
# for use by other scripts (like the API server and teardown script).
DEPLOYED_INDEX_NAME="scribe_v3_deployed_streaming"
python -c "from utils import update_env_file; update_env_file('VERTEX_DEPLOYED_INDEX_ID', '$DEPLOYED_INDEX_NAME')"
# Re-source one last time to get the deployed index ID
source ./startup.sh

echo "--- DEPLOYING NEW INDEX (This can take 20-60 minutes) ---"
gcloud ai index-endpoints deploy-index "${VERTEX_ENDPOINT_ID}" \
  --index="${VERTEX_INDEX_ID}" \
  --display-name="scribe-v3-deployed-index-streaming" \
  --deployed-index-id="${VERTEX_DEPLOYED_INDEX_ID}" \
  --project="${GOOGLE_CLOUD_PROJECT}" \
  --region="${GOOGLE_CLOUD_LOCATION}" \
  --quiet

echo "--- Deployment Submitted. ---"
echo "The Index is being deployed to the Endpoint. This can take up to an hour."
echo "You can check the status in the Google Cloud Console or by using 'gcloud ai operations describe'."