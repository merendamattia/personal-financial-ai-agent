# Financial AI Agent
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py .
COPY src/ src/
COPY prompts/ prompts/
COPY dataset/ dataset/
COPY .env.example .env

# Unzip dataset
RUN cd dataset && unzip ETFs.zip && rm ETFs.zip

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Configure Streamlit
RUN mkdir -p ~/.streamlit && \
    echo "[client]" > ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "port = 8501" >> ~/.streamlit/config.toml && \
    echo "[logger]" >> ~/.streamlit/config.toml && \
    echo "level = \"info\"" >> ~/.streamlit/config.toml

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
