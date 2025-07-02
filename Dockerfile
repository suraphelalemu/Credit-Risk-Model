FROM python:3.-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build tools and Python headers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set MLflow tracking URI (can be overridden at runtime)
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
