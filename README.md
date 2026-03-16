# Brainrot Detector API

A multimodal machine learning API system for detecting "brainrot" content in videos. The system uses a complete fusion approach, combining visual, audio, and text (transcript) features to accurately classify content.

## Features

- **Multimodal Fusion Engine**: Analyzes three distinct modalities:
  - **Visual**: Frames processed via EfficientNet-B0.
  - **Audio**: Mel spectrograms processed into DB scale.
  - **Text**: Whisper transcripts encoded via DistilBERT.
- **FastAPI Server**: Provides standard inference and ensemble endpoints.
- **Local Prediction Script**: Easily classify videos via command line.
- **Model Checkpoint Auto-Loading**: Fast inference with pre-loaded models.
- **Ensemble Support**: Combine predictions from all trained folds for maximum accuracy.

## Prerequisites

Before running the application, make sure your system has the following installed:
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) (Mandatory for audio/video extraction)

## Installation

1. **Clone the repository** (or extract the project files).
2. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn python-multipart torch torchvision opencv-python librosa transformers==4.40.0 tensorflow openai-whisper h5py numpy
   ```
   *Note: Using a virtual environment (`venv` or `conda`) is highly recommended.*

3. **Checkpoints**: Ensure your trained model `.pt` files (e.g., `BEST_fold1.pt`) are stored in the `./checkpoints` directory relative to the scripts.

## Usage: API Server

The project includes a fully featured REST API built with FastAPI.

**Start the Server:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Once running, access the auto-generated Swagger UI documentation at `http://localhost:8000/docs`.

**Endpoints:**
- `GET /health` : Verify server health and see loaded fold checkpoints.
- `GET /model/info` : Get technical details about model feature extraction dimensions.
- `POST /predict` : Fast classification using the best available single fold model.
- `POST /predict/ensemble` : High-accuracy classification by averaging predictions across all available fold checkpoints.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@test_video.mp4"
```

## Usage: Local Inference Script

You can also run predictions on videos locally without starting the API server using `predict.py`.

**Quick Start (Best Single Fold):**
```bash
python predict.py --video path/to/your_video.mp4
```

**Use a Specific Checkpoint:**
```bash
python predict.py --video path/to/video.mp4 --model checkpoints/BEST_fold2.pt
```

**Run Full Ensemble (Maximum Accuracy):**
```bash
python predict.py --video path/to/video.mp4 --all_folds
```

## Project Structure

- `api_server.py`: The FastAPI application server handling requests.
- `predict.py`: CLI tool for local inference.
- `brainrot-detector/`: Frontend application files (HTML, JS, CSS) to interact with the API interface.
- `checkpoints/`: Directory storing trained PyTorch model weights (`.pt`).
