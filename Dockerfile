# Use Python 3.10 slim image
FROM python:3.10-slim

# Set up the working directory
WORKDIR /app

# Install system dependencies required for OpenCV, librosa, yt-dlp, and ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running the app as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt /app/

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend files (api_server.py, backend_api, model_checkpoints)
COPY --chown=user . /app/

# Expose port 7860 as expected by Hugging Face Spaces
EXPOSE 7860

# Start the FastAPI app via Uvicorn on port 7860
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
