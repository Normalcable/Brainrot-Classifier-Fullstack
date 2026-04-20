# Brainrot Detector API

A multimodal machine learning API system for detecting "brainrot" content in videos. The system uses a complete fusion approach, combining visual, audio, and text (transcript) features to accurately classify content.

## Features

- **Multimodal Fusion Engine**: Analyzes three distinct modalities:
  - **Visual**: Frames processed via EfficientNet-B0.
  - **Audio**: Mel spectrograms processed into DB scale.
  - **Text**: Whisper transcripts encoded via DistilBERT.
- **Multi-Model Version Support**: Switch between trained model versions at runtime:
  - `default` — Full dataset including YouTube brainrot content.
  - `no_yt` — Trained without YouTube brainrot dataset (stored in `./checkpoint2`).
- **FastAPI Server**: Provides standard inference, ensemble, and model-switching endpoints.
- **Local Prediction Script**: Easily classify videos via command line with `--model_version`.
- **Swap-on-Demand**: Only one version's fold weights are in memory at a time; feature extractors (EfficientNet, DistilBERT) stay loaded across switches.
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
- `GET  /health`          : Verify server health, see loaded folds and active model version.
- `GET  /model/info`      : Get technical details about the active model configuration.
- `GET  /models`          : List all available model versions and which is active.
- `POST /models/switch?version=no_yt` : Hot-swap the active model version (keeps feature extractors loaded).
- `POST /predict?model_version=default`   : Fast classification using a single fold. Auto-swaps version if needed.
- `POST /predict/ensemble?model_version=no_yt` : High-accuracy ensemble. Auto-swaps version if needed.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@test_video.mp4"
```

## Model Versions

| Version ID | Description | Checkpoint Dir |
|---|---|---|
| `default` | Full dataset (includes YouTube brainrot) | `./checkpoints` |
| `no_yt` | No YouTube brainrot dataset | `./checkpoint2` |

## Usage: Local Inference Script

You can also run predictions on videos locally without starting the API server using `predict.py`.

**Quick Start (Default model version):**
```bash
python predict.py --video path/to/your_video.mp4
```

**Use the No-YouTube model version:**
```bash
python predict.py --video path/to/video.mp4 --model_version no_yt
```

**Run Full Ensemble with a specific version:**
```bash
python predict.py --video path/to/video.mp4 --model_version no_yt --all_folds
```

**Use a Specific Checkpoint file:**
```bash
python predict.py --video path/to/video.mp4 --model checkpoints/BEST_fold2.pt
```

**List all available model versions:**
```bash
python predict.py --help
```

## Project Structure

- `api_server.py`: The FastAPI application server handling requests and model version management.
- `predict.py`: CLI tool for local inference with `--model_version` support.
- `frontend_web/`: Frontend application (HTML, JS, CSS) with model version selector UI.
- `backend_api/`: The backend components and source for the FastAPI server (previously brainrot_identifier).
- `checkpoints/`: Default model weights — trained on full dataset including YouTube.
- `checkpoint2/`: Alternative model weights — trained without YouTube brainrot dataset (`no_yt`).
