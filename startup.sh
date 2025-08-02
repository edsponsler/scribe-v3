#!/bin/bash

# This script sets the necessary environment variables for Google Cloud.
# To apply these variables to your current shell session, you must run
# this script using the 'source' command:
#
#   source ./startup.sh
#
# Or using the dot operator:
#
#   . ./startup.sh

echo "Setting Google Cloud environment variables..."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Please create it from .env-example."
fi

echo "✓ GOOGLE_CLOUD_PROJECT set to: ${GOOGLE_CLOUD_PROJECT}"
echo "✓ GOOGLE_CLOUD_LOCATION set to: ${GOOGLE_CLOUD_LOCATION}"
echo ""
echo "Configuring gcloud CLI and Application Default Credentials (ADC)..."

# Set the active project for the gcloud CLI.
gcloud config set project "${GOOGLE_CLOUD_PROJECT}"

# Set the quota project for Application Default Credentials (ADC) to match.
gcloud auth application-default set-quota-project "${GOOGLE_CLOUD_PROJECT}"
