#!/bin/bash
echo "--- TEARING DOWN ALL EXPENSIVE CLOUD RESOURCES ---"
# Read IDs from the .env file
INDEX_ID=$(grep VERTEX_INDEX_ID .env | cut -d '=' -f2)
ENDPOINT_ID=$(grep VERTEX_ENDPOINT_ID .env | cut -d '=' -f2)
DEPLOYED_ID=$(grep VERTEX_DEPLOYED_INDEX_ID .env | cut -d '=' -f2)
PROJECT_ID=$(grep GOOGLE_CLOUD_PROJECT .env | cut -d '=' -f2)
REGION=$(grep GOOGLE_CLOUD_LOCATION .env | cut -d '=' -f2)

echo "--- Undeploying Index (This can take 10-20 minutes) ---"
gcloud ai index-endpoints undeploy-index $ENDPOINT_ID \
  --deployed-index-id=$DEPLOYED_ID \
  --project=$PROJECT_ID \
  --region=$REGION \
  --quiet

echo "--- Deleting Index ---"
gcloud ai indexes delete $INDEX_ID \
  --project=$PROJECT_ID \
  --region=$REGION \
  --quiet

# --- START: Corrected Step ---
echo "--- Deleting Index Endpoint (This stops all hourly charges) ---"
gcloud ai index-endpoints delete $ENDPOINT_ID \
  --project=$PROJECT_ID \
  --region=$REGION \
  --quiet
# --- END: Corrected Step ---

echo "--- Teardown Complete. All hourly charges have been stopped. ---"