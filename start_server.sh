#!/bin/bash

# Check for required environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | sed 's/#.*//g' | xargs)
fi

# Check for required environment variables
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
  echo "Error: GOOGLE_CLOUD_PROJECT environment variable is not set."
  exit 1
fi

if [ -z "$GOOGLE_CLOUD_LOCATION" ]; then
  echo "Error: GOOGLE_CLOUD_LOCATION environment variable is not set."
  exit 1
fi

if [ -z "$VERTEX_GENERATIVE_MODEL" ]; then
  echo "Error: VERTEX_GENERATIVE_MODEL environment variable is not set."
  exit 1
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8080
