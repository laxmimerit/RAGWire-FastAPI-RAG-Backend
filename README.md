# RAGWire - FastAPI RAG Backend
**Deploy Your Agent in Production at Render, Railway, AWS, GCP, and Azure.**

An OpenAI-compatible FastAPI server powered by [RAGWire](https://kgptalkie.com/ragwire-api-reference/), supporting multiple agent frameworks (LangChain, LangGraph, CrewAI, AutoGen) with Qdrant vector store and Google Gemini.

---

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

AGENT=01_langchain_agent   # see Available Agents below

LANGCHAIN_TRACING_V2=false
CREWAI_TRACING_ENABLED=false
```

### Available Agents

| Value | Agent |
|---|---|
| `01_langchain_agent` | LangChain (default) |
| `02_langgraph_self_correcting_agent` | LangGraph self-correcting |
| `03_langgraph_supervisor_agent` | LangGraph supervisor |
| `04_crewai_agent` | CrewAI single agent |
| `05_crewai_multiagent` | CrewAI multi-agent |
| `06_autogen_agent` | AutoGen single agent |
| `07_microsoft_agent` | Microsoft Agent Framework |
| `08_microsoft_multiagent` | Microsoft Multi-agent |

---

## Docker

### Run locally

```bash
# Build the image
docker build -t fastapi-rag-backend .

# Run the container
docker run -p 8080:8080 --env-file .env fastapi-rag-backend
```

Server runs at `http://localhost:8080`

---

## Deployment

### Railway

1. Push code to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select your repository — Railway auto-detects the `Dockerfile`
4. Add environment variables under **Variables** tab
5. Go to **Settings → Networking → Generate Domain**

### Render

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repository — Render auto-detects the `Dockerfile`
3. Add environment variables under **Environment** tab
4. Click **Deploy**

> **Note:** Free tier spins down after 15 min of inactivity.

### AWS ECS Express Mode

> App Runner no longer accepts new customers as of April 30, 2026. AWS recommends **Amazon ECS Express Mode** for containerized deployments.

#### 1. Install AWS CLI

```bash
# macOS
brew install awscli

# Windows
winget install Amazon.AWSCLI

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
```

#### 2. Configure credentials

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, region (e.g. us-east-1), output format (json)
```

#### 3. Push image to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name fastapi-rag-backend

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 783330586114.dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -t fastapi-rag-backend .
docker tag fastapi-rag-backend:latest 783330586114.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag-backend:latest
docker push 783330586114.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag-backend:latest
```

#### 4. Deploy with ECS Express Mode

1. Go to **AWS Console → Elastic Container Service → Services → Create**
2. Select **Express** mode
3. Paste your ECR image URI:
   ```
   783330586114.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag-backend:latest
   ```
4. Set container port to `8080`
5. Add environment variables (GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY etc.)
6. Let AWS auto-create the required IAM roles when prompted
7. Click **Create** — AWS automatically provisions load balancer, networking, HTTPS endpoint, and auto-scaling

Your app will be live at the auto-generated HTTPS URL shown in the console.

---

### GCP Cloud Run

#### 1. Install gcloud CLI

```bash
# macOS
brew install --cask google-cloud-sdk

# Windows (winget)
winget install Google.CloudSDK

# Windows (manual) - download installer from:
# https://cloud.google.com/sdk/docs/install

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

Verify installation:
```bash
gcloud --version
```

---

#### 2. Initialize and login

```bash
# Login to your Google account
gcloud auth login

# Initialize gcloud (select project, region etc.)
gcloud init
```

> If you don't have a GCP project yet: go to **console.cloud.google.com → New Project** → copy the Project ID.

---

#### 3. Set your project

```bash
gcloud config set project <your-project-id>

# Verify
gcloud config get project
```

---

#### 4. Enable required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

---

#### 5. Authenticate Docker to Google Cloud

```bash
gcloud auth configure-docker
```

---

#### 6. Deploy to Cloud Run

Cloud Run builds and deploys the Docker image automatically from source — no manual `docker build` or `docker push` needed:

First fill in your values in `env.yaml`, then deploy:

```bash
gcloud run deploy fastapi-rag-backend --source . --region us-central1 --allow-unauthenticated --port 8080 --env-vars-file env.yaml
```

> Cloud Run uses **Cloud Build** to build your image and stores it in **Artifact Registry** automatically.

---

#### 7. Get your public URL

After deploy, the URL is printed in the terminal:
```
Service URL: https://fastapi-rag-backend-xxxxxxxxxx-uc.a.run.app
```

Verify:
```bash
curl https://fastapi-rag-backend-xxxxxxxxxx-uc.a.run.app/health
# Expected: {"status": "ok"}
```

---

#### 8. Update deployment (after code changes)

Just run the same deploy command again — Cloud Run rebuilds and redeploys with zero downtime:
```bash
gcloud run deploy fastapi-rag-backend --source . --region us-central1
```

---

### Azure Container Apps

#### 1. Install Azure CLI

```bash
# macOS
brew install azure-cli

# Windows
winget install Microsoft.AzureCLI

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

#### 2. Login and create resource group

```bash
az login
az group create --name ragwire-rg --location eastus
```

#### 3. Create Azure Container Registry and push image

```bash
# Create ACR
az acr create --name ragwireacr --resource-group ragwire-rg --sku Basic

# Login to ACR
az acr login --name ragwireacr

# Build and push image
docker build -t ragwireacr.azurecr.io/fastapi-rag-backend:latest .
docker push ragwireacr.azurecr.io/fastapi-rag-backend:latest
```

> **Note:** `az containerapp up --source .` uses ACR Tasks to build the image, which is not available on free/trial Azure subscriptions. Building and pushing locally bypasses this restriction.

#### 4. Deploy

Run locally **or** from [Azure Cloud Shell](https://portal.azure.com) (recommended if you have TLS/network issues locally):

```bash
az containerapp up \
  --name fastapi-rag-backend \
  --image ragwireacr.azurecr.io/fastapi-rag-backend:latest \
  --resource-group ragwire-rg \
  --ingress external \
  --target-port 8080
```

#### 5. Set environment variables

**macOS / Ubuntu / Azure Cloud Shell** — reads directly from `.env` file:

```bash
az containerapp update --name fastapi-rag-backend \
  --resource-group ragwire-rg \
  --set-env-vars $(grep -v '^#' .env | grep '=' | xargs)
```

**Windows (PowerShell)**:

```powershell
$envVars = (Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' })

az containerapp update --name fastapi-rag-backend --resource-group kgptalkie_rg_8216 --set-env-vars @envVars
```

---

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat with the RAG agent (streaming) |
| POST | `/upload` | Upload documents for ingestion |

---

## OpenWebUI Integration

1. Go to OpenWebUI → **Settings → Connections**
2. Set **URL** to your deployed API URL (e.g. `https://your-app.up.railway.app`)
3. Select the model and start chatting

## Upload Documents

```bash
curl -X POST https://your-api-url/upload -F "files=@document.pdf" -F "files=@report.docx"
```

---

## Authentication

The server supports optional API key authentication via Bearer token (same as OpenAI).

**1. Add to `routes.py`:**

```python
import os
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

API_KEY = os.getenv("API_KEY")
bearer = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(bearer)):
    if API_KEY and (not credentials or credentials.credentials != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

**2. Add `dependencies=[Depends(verify_api_key)]` to each route you want to protect:**

```python
@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
@router.post("/upload", dependencies=[Depends(verify_api_key)])
```

**3. Set the env var:**

```env
API_KEY=your-secret-key
```

> If `API_KEY` is not set, authentication is disabled and the API is open. The `/health` endpoint is always public.
