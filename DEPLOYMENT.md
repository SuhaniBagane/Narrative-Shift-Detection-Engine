# BuzzStreet Deployment Guide

## Docker Container Deployment

```bash
# Build production Docker image
docker build -t buzzstreet-engine:latest .

# Run standalone container
docker run -d -p 8501:8501 --name buzzstreet_app buzzstreet-engine:latest

# Or run with Docker Compose
docker-compose up --build -d
```

## Production Verification & Health Checks
- **Streamlit Web Portal:** `http://localhost:8501`
- **Docker Container Status:** `docker ps`
- **Container Logs:** `docker logs -f buzzstreet_app`
