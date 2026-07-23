# Build stage using official Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for compiling Stan (for prophet) or LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements list first to cache dependency layers
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Ensure test libraries are installed
    pip install pytest pytest-cov

# Copy the entire project code into the container
COPY . /app/

# Expose Streamlit default port
EXPOSE 8501

# Run synthetic data generation to initialize testing database sandbox
RUN python scripts/download_data.py --sample

# Command to run the Streamlit dashboard app
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
