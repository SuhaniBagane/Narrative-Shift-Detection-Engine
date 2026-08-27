# BuzzStreet – Production Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir twilio python-dotenv scipy plotly pandas numpy scikit-learn nltk streamlit

# Copy application files
COPY . .

# Download NLTK data dependencies
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
