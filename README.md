# Scribe v3

This is the starting point for the Scribe v3 project. This document outlines the necessary steps to set up your local development environment and configure the required Google Cloud services.

## Getting Started

Follow these instructions to get your development environment up and running.

### Prerequisites

*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (gcloud CLI) installed and authenticated.
*   A Google Cloud Project.
*   Python 3.12+

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd scribe-v3
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    # Install dependencies for the ingestion scripts
    pip install -r ingestion/requirements.txt
    ```

4.  **Configure Environment Variables:**
    This project uses a `startup.sh` script to configure both shell environment variables and `gcloud` settings. It reads values from the `.env` file.

    An example file, `.env-example`, is provided. Copy it to create your own `.env` file:
    ```bash
    cp .env-example .env
    ```

    Next, open the `.env` file and replace the placeholder values with your specific Google Cloud Project ID, location, and Cloud Storage bucket name.

    Then, run the startup script using `source` to apply the settings to your current terminal session. You will need to do this for every new session.
    ```bash
    source ./startup.sh
    ```
    This script will:
    *   Export `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` as environment variables.
    *   Set the active `gcloud` project.
    *   Set the Application Default Credentials (ADC) quota project to match.

### Enable Google Cloud APIs

This project relies on several Google Cloud services. You must enable their APIs in your project before you can use them.

Run the following command to enable all the necessary APIs. The command uses the `$GOOGLE_CLOUD_PROJECT` environment variable, which is set by the `startup.sh` script.

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  documentai.googleapis.com \
  bigquery.googleapis.com --project=${GOOGLE_CLOUD_PROJECT}
```

This will enable the following services:

*   **Vertex AI API** (`aiplatform.googleapis.com`): For accessing generative AI models like Gemini.
*   **Cloud Storage API** (`storage.googleapis.com`): For storing and retrieving project files and assets.
*   **Cloud Run API** (`run.googleapis.com`): For deploying and managing containerized applications.
*   **Document AI API** (`documentai.googleapis.com`): For processing and extracting data from documents.
*   **BigQuery API** (`bigquery.googleapis.com`): For data analysis and warehousing.

Once these steps are complete, your environment is ready for development.

### Create a Service Account

For applications to authenticate with Google Cloud services programmatically (e.g., from a CI/CD pipeline or a deployed service like Cloud Run), it is best practice to use a dedicated service account instead of personal user credentials.

The following commands will create a service account and grant it the necessary permissions for this project. These commands use the `$GOOGLE_CLOUD_PROJECT` environment variable set by the `startup.sh` script.

1.  **Define a name for your service account and create it:**
    ```bash
    export SA_NAME="scribe-programmatic-user"

    gcloud iam service-accounts create ${SA_NAME} \
      --description="Service account for SCRIBE v3 application" \
      --display-name="SCRIBE Programmatic User" \
      --project=${GOOGLE_CLOUD_PROJECT}
    ```

2.  **Grant the necessary IAM roles to the service account:**
    The service account needs permissions to interact with Vertex AI (for Gemini) and Cloud Storage.
    ```bash
    # Grant Vertex AI User role
    gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
      --member="serviceAccount:${SA_NAME}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com" \
      --role="roles/aiplatform.user"

    # Grant Storage Object Admin role
    gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
      --member="serviceAccount:${SA_NAME}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com" \
      --role="roles/storage.objectAdmin"
    ```

### Create Cloud Storage Bucket

This project requires a Cloud Storage bucket to act as a data lake for storing raw, unprocessed data. The following command will create a bucket for this purpose. It uses the `$GOOGLE_CLOUD_PROJECT` environment variable to help ensure the bucket name is globally unique.

**Important**: The name of the bucket created here must match the `GCS_BUCKET_NAME` value in your `.env` file.

After setting `GCS_BUCKET_NAME` in your `.env` file and running `source ./startup.sh`, create the bucket with the following command:

```bash
gcloud storage buckets create gs://${GCS_BUCKET_NAME} \
  --project=${GOOGLE_CLOUD_PROJECT} \
  --location=${GOOGLE_CLOUD_LOCATION}
```

### Setup Vertex AI Vector Search

To enable efficient similarity searches on the ingested texts, this project uses [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview). The setup involves creating a vector index to store the embeddings and deploying it to a queryable endpoint.

The data ingestion pipeline has also been updated:
*   `ingestion/text_processor.py`: This new module is responsible for cleaning the raw text from Project Gutenberg and splitting it into manageable chunks (paragraphs) before embeddings are generated.
*   `ingestion/gutenberg_harvester.py`: This script now uses the text processor and the Vertex AI SDK to generate embeddings for the text chunks.

Follow these steps to create and deploy the index.

#### 1. Create the Vector Index 

The create_index.py script creates a new Brute Force index in Vertex AI. After creation, it will automatically update your .env file with the new VERTEX_INDEX_ID.

```bash
python create_index.py
```

#### 2. Create the Index Endpoint

An Index Endpoint is the running, scalable server that loads your index into memory. The create_endpoint.py script creates a public endpoint and automatically updates your .env file with the new VERTEX_ENDPOINT_ID.

```bash
python create_endpoint.py
```

#### 3. Deploy the Index to the Endpoint

Finally, deploy the index to the endpoint to make it queryable. This command uses the IDs that were saved to your .env file and loaded by startup.sh.
Note: This deployment process can take 20-60 minutes to complete.

```bash
# Make sure you have run 'source ./startup.sh' first
gcloud ai index-endpoints deploy-index ${VERTEX_ENDPOINT_ID} \
  --index=${VERTEX_INDEX_ID} \
  --display-name="scribe-v3-deployed-index-streaming" \
  --deployed-index-id=${VERTEX_DEPLOYED_INDEX_ID} \
  --project=${GOOGLE_CLOUD_PROJECT} \
  --region=${GOOGLE_CLOUD_LOCATION}
  ```

You can check the status of this process using the 'gcloud ai operations describe' command provided in the output of the previous command.

## Running the Data Ingestion

Once all the setup steps are complete and the index has been successfully deployed to the endpoint, you can begin ingesting content. 

The ingestion/gutenberg_harvester.py script orchestrates this process. It reads the book IDs from gutenberg_manifest.txt, downloads each book, cleans and chunks the text, generates embeddings using the Vertex AI text-embedding-004 model, and finally upserts the embeddings into your deployed Vector Search index.

To run the ingestion, execute the following command from the project root:

```bash
python -m ingestion.gutenberg_harvester
```

> **Note on Ingestion Limits**
> For testing purposes, the script is currently configured to process only the *first book* from the manifest and only the *first 5 text chunks* from that book. You can modify the `main()` and `process_book()` functions in `gutenberg_harvester.py` to ingest all content.
