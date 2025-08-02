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
    # Make sure you have a requirements.txt file
    pip install -r requirements.txt
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
gcloud services enable aiplatform.googleapis.com storage.googleapis.com run.googleapis.com documentai.googleapis.com bigquery.googleapis.com --project=${GOOGLE_CLOUD_PROJECT}
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

```bash
export BUCKET_NAME="scribe-v3-raw-data-lake-${GOOGLE_CLOUD_PROJECT}"
gcloud storage buckets create gs://${BUCKET_NAME} \
  --project=${GOOGLE_CLOUD_PROJECT} \
  --location=US
```
