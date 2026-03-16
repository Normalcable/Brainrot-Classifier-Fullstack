"""
=============================================================================
BRAINROT DETECTOR — FASTAPI SERVER
=============================================================================
Setup:
    pip install fastapi uvicorn python-multipart torch torchvision
    pip install opencv-python librosa transformers==4.40.0 tensorflow

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

API Docs (auto-generated):
    http://localhost:8000/docs

Endpoints:
    POST /predict          — classify a single uploaded video
    POST /predict/ensemble — classify using all fold models (more accurate)
    GET  /health           — check server status
    GET  /model/info       — get loaded model info
=============================================================================
"""

import os
import gc
import uuid
import time
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# ── CONFIG — update to match your setup ─────────────────────────────────────
CHECKPOINT_DIR  = "./checkpoints"
N_FOLDS         = 3
VISUAL_FEAT_DIM = 1280
AUDIO_FEAT_DIM  = 2048
TEXT_FEAT_DIM   = 768
FUSION_DIM      = 256
NUM_CLASSES     = 2
MAX_TEXT_LEN    = 128
N_MELS          = 128
AUDIO_MAX_FRAMES= 300
IMG_SIZE        = 224
MAX_FRAMES      = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# MODEL ARCHITECTURE (must match training)
# =============================================================================

class ModalityProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


class AttentionFusion(nn.Module):
    def __init__(self, fusion_dim, num_modalities=3):
        super().__init__()
        self.attn = nn.Linear(fusion_dim * num_modalities, num_modalities)

    def forward(self, feats):
        cat     = torch.cat(feats, dim=-1)
        weights = torch.softmax(self.attn(cat), dim=-1)
        stacked = torch.stack(feats, dim=1)
        fused   = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return fused


class BrainrotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_proj = ModalityProjector(VISUAL_FEAT_DIM, FUSION_DIM)
        self.audio_proj  = ModalityProjector(AUDIO_FEAT_DIM,  FUSION_DIM)
        self.text_proj   = ModalityProjector(TEXT_FEAT_DIM,   FUSION_DIM)
        self.fusion      = AttentionFusion(FUSION_DIM, num_modalities=3)
        self.classifier  = nn.Sequential(
            nn.Linear(FUSION_DIM, FUSION_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_DIM // 2, NUM_CLASSES),
        )

    def forward(self, visual, audio, text):
        v = self.visual_proj(visual)
        a = self.audio_proj(audio)
        t = self.text_proj(text)
        return self.classifier(self.fusion([v, a, t]))


# =============================================================================
# MODEL MANAGER (loads once on startup, reuses for every request)
# =============================================================================

class ModelManager:
    def __init__(self):
        self.models     = {}
        self.eff_model  = None
        self.bert_model = None
        self.tokenizer  = None
        self._loaded    = False

    def load_all(self):
        if self._loaded:
            return

        print(f"\n[ModelManager] Loading models on {DEVICE}...")

        # Load PyTorch fold models
        for fold_num in range(1, N_FOLDS + 1):
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"BEST_fold{fold_num}.pt")
            if os.path.exists(ckpt_path):
                model = BrainrotModel().to(DEVICE)
                ckpt  = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt["model"])
                model.eval()
                self.models[fold_num] = model
                print(f"  [✓] Loaded BEST_fold{fold_num}.pt")
            else:
                print(f"  [!] Not found: {ckpt_path}")

        if not self.models:
            raise RuntimeError(f"No checkpoints found in: {CHECKPOINT_DIR}")

        # Load TF feature extractors
        import tensorflow as tf
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras import mixed_precision
        from transformers import TFDistilBertModel, DistilBertTokenizer

        mixed_precision.set_global_policy('float32')

        print("  [✓] Loading EfficientNet-B0...")
        self.eff_model = EfficientNetB0(
            include_top=False, weights='imagenet', pooling='avg'
        )
        self.eff_model.trainable = False

        print("  [✓] Loading DistilBERT...")
        self.tokenizer  = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert_model.trainable = False

        self._loaded = True
        print(f"[ModelManager] All models loaded. Folds available: {list(self.models.keys())}\n")

    @property
    def available_folds(self):
        return list(self.models.keys())


model_manager = ModelManager()


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_visual(video_path: str) -> np.ndarray:
    import cv2

    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    indices = np.linspace(0, max(total - 1, 0), MAX_FRAMES, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32))
    cap.release()

    if not frames:
        return np.zeros((1, VISUAL_FEAT_DIM), dtype=np.float32)

    mean_frame = np.mean(np.array(frames), axis=0)[None]
    feats = model_manager.eff_model.predict(mean_frame, verbose=0)
    return feats.astype(np.float32)


def extract_audio(video_path: str) -> np.ndarray:
    import librosa

    audio_path = video_path + "_audio.wav"
    ret = os.system(f'ffmpeg -i "{video_path}" -ac 1 -ar 22050 "{audio_path}" -y -loglevel quiet')

    if ret != 0 or not os.path.exists(audio_path):
        return np.zeros((1, AUDIO_FEAT_DIM), dtype=np.float32)

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        os.remove(audio_path)

        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32).T

        if mel_db.shape[0] >= AUDIO_MAX_FRAMES:
            mel_db = mel_db[:AUDIO_MAX_FRAMES, :]
        else:
            pad    = np.zeros((AUDIO_MAX_FRAMES - mel_db.shape[0], N_MELS), dtype=np.float32)
            mel_db = np.vstack([mel_db, pad])

        flat = mel_db.flatten()[None]
        if flat.shape[1] >= AUDIO_FEAT_DIM:
            return flat[:, :AUDIO_FEAT_DIM].astype(np.float32)
        else:
            feat = np.zeros((1, AUDIO_FEAT_DIM), dtype=np.float32)
            feat[:, :flat.shape[1]] = flat
            return feat
    except Exception:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return np.zeros((1, AUDIO_FEAT_DIM), dtype=np.float32)


def extract_text(video_path: str) -> tuple:
    """Returns (feature_array, transcript_string)"""
    transcript = "no speech detected"

    try:
        import whisper
        audio_path = video_path + "_whisper.wav"
        os.system(f'ffmpeg -i "{video_path}" -ac 1 -ar 16000 "{audio_path}" -y -loglevel quiet')
        if os.path.exists(audio_path):
            wm         = whisper.load_model("tiny")
            result     = wm.transcribe(audio_path)
            transcript = result.get("text", "").strip() or "no speech detected"
            os.remove(audio_path)
            del wm
            gc.collect()
    except ImportError:
        pass

    encoded = model_manager.tokenizer(
        [transcript],
        padding='max_length',
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors='np'
    )
    output = model_manager.bert_model(
        input_ids=encoded['input_ids'].astype(np.int32),
        attention_mask=encoded['attention_mask'].astype(np.int32)
    )
    feat = output.last_hidden_state[:, 0, :].numpy()
    return feat.astype(np.float32), transcript


def run_inference(video_path: str, fold_nums: Optional[List[int]] = None) -> dict:
    """Core inference logic — used by both single and ensemble endpoints."""
    if fold_nums is None:
        fold_nums = [min(model_manager.available_folds)]

    # Extract features once, reuse across folds
    vis_feat            = extract_visual(video_path)
    aud_feat            = extract_audio(video_path)
    txt_feat, transcript = extract_text(video_path)

    vis_t = torch.tensor(vis_feat, dtype=torch.float32).to(DEVICE)
    aud_t = torch.tensor(aud_feat, dtype=torch.float32).to(DEVICE)
    txt_t = torch.tensor(txt_feat, dtype=torch.float32).to(DEVICE)

    per_fold_probs = []
    for fold_num in fold_nums:
        if fold_num not in model_manager.models:
            continue
        model = model_manager.models[fold_num]
        with torch.no_grad():
            logits = model(vis_t, aud_t, txt_t)
            probs  = torch.softmax(logits, dim=-1)
        per_fold_probs.append(float(probs[0][1].cpu()))

    if not per_fold_probs:
        raise HTTPException(status_code=500, detail="No valid fold models available.")

    avg_prob        = float(np.mean(per_fold_probs))
    prediction      = "BRAINROT" if avg_prob >= 0.5 else "NON_BRAINROT"
    confidence      = avg_prob if prediction == "BRAINROT" else 1 - avg_prob

    return {
        "prediction":        prediction,
        "confidence":        round(confidence, 4),
        "prob_brainrot":     round(avg_prob, 4),
        "prob_non_brainrot": round(1 - avg_prob, 4),
        "transcript":        transcript,
        "folds_used":        len(per_fold_probs),
        "per_fold_probs":    [round(p, 4) for p in per_fold_probs],
    }


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Brainrot Detector API",
    description="Multimodal brainrot video classification using Visual + Audio + Text fusion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    video_name:        str
    prediction:        str
    confidence:        float
    prob_brainrot:     float
    prob_non_brainrot: float
    transcript:        str
    folds_used:        int
    per_fold_probs:    List[float]
    processing_time_s: float


class HealthResponse(BaseModel):
    status:         str
    device:         str
    folds_loaded:   List[int]
    models_ready:   bool


class ModelInfoResponse(BaseModel):
    checkpoint_dir: str
    folds_available: List[int]
    device:         str
    visual_dim:     int
    audio_dim:      int
    text_dim:       int
    fusion_dim:     int


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("[API] Starting up — loading models...")
    model_manager.load_all()
    print("[API] Ready to serve requests.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check():
    """Check if the API server and models are ready."""
    return HealthResponse(
        status       = "ok",
        device       = str(DEVICE),
        folds_loaded = model_manager.available_folds,
        models_ready = model_manager._loaded,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Status"])
async def model_info():
    """Get information about the loaded model configuration."""
    return ModelInfoResponse(
        checkpoint_dir   = CHECKPOINT_DIR,
        folds_available  = model_manager.available_folds,
        device           = str(DEVICE),
        visual_dim       = VISUAL_FEAT_DIM,
        audio_dim        = AUDIO_FEAT_DIM,
        text_dim         = TEXT_FEAT_DIM,
        fusion_dim       = FUSION_DIM,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    video: UploadFile = File(..., description="Video file to classify (mp4, avi, mov)")
):
    """
    Classify a video using the best model from Fold 1.
    Faster than ensemble — good for quick classification.
    """
    if not model_manager._loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}"
        )

    # Save uploaded file to temp location
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            content = await video.read()
            f.write(content)

        start_time = time.time()
        result     = run_inference(tmp_path, fold_nums=[min(model_manager.available_folds)])
        elapsed    = time.time() - start_time

        return PredictionResponse(
            video_name        = video.filename,
            processing_time_s = round(elapsed, 2),
            **result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict/ensemble", response_model=PredictionResponse, tags=["Inference"])
async def predict_ensemble(
    video: UploadFile = File(..., description="Video file to classify (mp4, avi, mov)")
):
    """
    Classify a video using ALL fold models and average their predictions.
    More accurate than single-fold — recommended for final results.
    """
    if not model_manager._loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}"
        )

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            content = await video.read()
            f.write(content)

        start_time = time.time()
        result     = run_inference(tmp_path, fold_nums=model_manager.available_folds)
        elapsed    = time.time() - start_time

        return PredictionResponse(
            video_name        = video.filename,
            processing_time_s = round(elapsed, 2),
            **result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
