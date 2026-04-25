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
import re
import uuid
import time
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")

import psutil
import GPUtil
import asyncio
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# Video downloader engine
from video_downloader import get_scraper, detect_platform

# ── CONFIG — update to match your setup ─────────────────────────────────────
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

ACTIVE_TASKS_STATE = {}

def update_task_state(task_id: str, stage: str, log: str):
    if task_id:
        ACTIVE_TASKS_STATE[task_id] = {"stage": stage, "log": log}

# ── MODEL VERSIONS ──────────────────────────────────────────────────────────
MODEL_VERSIONS = {
    "default": {
        "name": "Default (Full Dataset)",
        "checkpoint_dir": "./model1_checkpoints",
        "n_folds": 3,
    },
    "no_yt": {
        "name": "No YouTube",
        "checkpoint_dir": "./model2_checkpoints",
        "n_folds": 3,
    },
}
DEFAULT_VERSION = "default"

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
        self.last_weights = weights
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
        self.models          = {}
        self.eff_model       = None
        self.bert_model      = None
        self.tokenizer       = None
        self._loaded         = False
        self.current_version = None

        # Image transforms matching EfficientNet V1 ImageNet training
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_feature_extractors(self):
        """Load shared feature extractors (only once, reused across versions)."""
        from transformers import DistilBertModel, DistilBertTokenizer

        print("  [OK] Loading EfficientNet-B0 (PyTorch)...")
        self.eff_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(DEVICE)
        self.eff_model.classifier = nn.Identity()  # Remove top layer to get features
        self.eff_model.eval()

        print("  [OK] Loading Wav2Vec2 (HuggingFace)...")
        self.tokenizer  = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE)
        self.bert_model.eval()

    def _load_fold_models(self, version_id: str):
        """Load fold checkpoint models for a specific version."""
        version_cfg    = MODEL_VERSIONS[version_id]
        checkpoint_dir = version_cfg["checkpoint_dir"]
        n_folds        = version_cfg["n_folds"]

        # Unload existing fold models
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[ModelManager] Loading version '{version_id}' ({version_cfg['name']}) from {checkpoint_dir}...")

        for fold_num in range(1, n_folds + 1):
            ckpt_path = os.path.join(checkpoint_dir, f"BEST_fold{fold_num}.pt")
            if os.path.exists(ckpt_path):
                model = BrainrotModel().to(DEVICE)
                ckpt  = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt["model"])
                model.eval()
                self.models[fold_num] = model
                print(f"  [OK] Loaded BEST_fold{fold_num}.pt")
            else:
                print(f"  [OK] Found checkpoint: {ckpt_path}")

        if not self.models:
            raise RuntimeError(f"No checkpoints found in: {checkpoint_dir}")

        self.current_version = version_id
        print(f"[ModelManager] Version '{version_id}' ready. Folds available: {list(self.models.keys())}")

    def load_all(self, version_id: str = None):
        """Initial load: feature extractors + default version fold models."""
        if version_id is None:
            version_id = DEFAULT_VERSION

        if not self._loaded:
            print("\n[ModelManager] Models loaded successfully!")
            self._load_feature_extractors()
            self._loaded = True

        self._load_fold_models(version_id)

    def switch_version(self, version_id: str):
        """Hot-swap to a different model version (keeps feature extractors loaded)."""
        if version_id not in MODEL_VERSIONS:
            raise ValueError(f"Unknown model version: '{version_id}'. Available: {list(MODEL_VERSIONS.keys())}")

        if version_id == self.current_version:
            print(f"[ModelManager] Version '{version_id}' is already loaded.")
            return

        self._load_fold_models(version_id)

    def ensure_version(self, version_id: str):
        """Ensure the requested version is loaded (auto-swap if needed)."""
        if version_id and version_id != self.current_version:
            self.switch_version(version_id)

    @property
    def available_folds(self):
        return list(self.models.keys())

    @property
    def version_name(self):
        if self.current_version and self.current_version in MODEL_VERSIONS:
            return MODEL_VERSIONS[self.current_version]["name"]
        return "Unknown"


model_manager = ModelManager()


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_visual(video_path: str, task_id: str = None) -> np.ndarray:
    import cv2
    update_task_state(task_id, "Extracting Visual", "[00:00] Initializing OpenCV visual sequence extraction...")

    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    indices = np.linspace(0, max(total - 1, 0), MAX_FRAMES, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixel_values = model_manager.img_transform(frame)
            frames.append(pixel_values)
    cap.release()

    if not frames:
        return np.zeros((1, VISUAL_FEAT_DIM), dtype=np.float32)

    batch = torch.stack(frames).to(DEVICE)
    with torch.no_grad():
        feats = model_manager.eff_model(batch) # (N, 1280)
    
    return feats.cpu().numpy().astype(np.float32)


def extract_audio(video_path: str, task_id: str = None) -> np.ndarray:
    import librosa
    update_task_state(task_id, "Extracting Audio", "[00:00] Running FFmpeg audio extraction to 22.05kHz...")

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


def extract_text(video_path: str, task_id: str = None) -> tuple:
    """Returns (feature_array, transcript_string)"""
    update_task_state(task_id, "Extracting Text", "[00:00] Running Whisper transcript generation...")
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
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        output = model_manager.bert_model(input_ids=input_ids, attention_mask=attention_mask)
    
    feat = output.last_hidden_state[:, 0, :].cpu().numpy()
    return feat.astype(np.float32), transcript


def run_inference(video_path: str, fold_nums: Optional[List[int]] = None, model_version: Optional[str] = None, task_id: str = None) -> dict:
    """Core inference logic — used by both single and ensemble endpoints."""
    update_task_state(task_id, "Starting Inference", f"Initializing run across {len(fold_nums) if fold_nums else 1} fold(s)...")
    # Auto-swap model version if requested
    if model_version:
        model_manager.ensure_version(model_version)

    if fold_nums is None:
        fold_nums = [min(model_manager.available_folds)]

    # Extract features once, reuse across folds
    t0 = time.time()
    vis_seq = extract_visual(video_path, task_id=task_id)
    vis_feat = np.mean(vis_seq, axis=0, keepdims=True)
    t1 = time.time()
    aud_feat = extract_audio(video_path, task_id=task_id)
    t2 = time.time()
    txt_feat, transcript = extract_text(video_path, task_id=task_id)
    t3 = time.time()

    update_task_state(task_id, "Fusing Modalities", "Computing final probabilities across folds...")
    vis_t = torch.tensor(vis_feat, dtype=torch.float32).to(DEVICE)
    aud_t = torch.tensor(aud_feat, dtype=torch.float32).to(DEVICE)
    txt_t = torch.tensor(txt_feat, dtype=torch.float32).to(DEVICE)

    per_fold_probs = []
    attentions_list = []
    
    for fold_num in fold_nums:
        if fold_num not in model_manager.models:
            continue
        model = model_manager.models[fold_num]
        with torch.no_grad():
            logits = model(vis_t, aud_t, txt_t)
            probs  = torch.softmax(logits, dim=-1)
            attentions_list.append(model.fusion.last_weights[0].cpu().numpy().tolist())
        per_fold_probs.append(float(probs[0][1].cpu()))

    if not per_fold_probs:
        raise HTTPException(status_code=500, detail="No valid fold models available.")

    temporal_probs = []
    first_fold = model_manager.models.get(fold_nums[0])
    if first_fold is not None:
        with torch.no_grad():
            vis_seq_t = torch.tensor(vis_seq, dtype=torch.float32).to(DEVICE)
            N = vis_seq_t.size(0)
            aud_rep = aud_t.repeat(N, 1)
            txt_rep = txt_t.repeat(N, 1)
            t_logits = first_fold(vis_seq_t, aud_rep, txt_rep)
            t_probs = torch.softmax(t_logits, dim=-1)[:, 1].cpu().numpy()
            temporal_probs = t_probs.tolist()

    t4 = time.time()

    avg_prob        = float(np.mean(per_fold_probs))
    prediction      = "BRAINROT" if avg_prob >= 0.5 else "NON_BRAINROT"
    confidence      = avg_prob if prediction == "BRAINROT" else 1 - avg_prob
    avg_attention = np.mean(attentions_list, axis=0).tolist() if attentions_list else [0.33, 0.33, 0.34]

    return {
        "prediction":        prediction,
        "confidence":        round(confidence, 4),
        "prob_brainrot":     round(avg_prob, 4),
        "prob_non_brainrot": round(1 - avg_prob, 4),
        "transcript":        transcript,
        "folds_used":        len(per_fold_probs),
        "per_fold_probs":    [round(p, 4) for p in per_fold_probs],
        "model_version":     model_manager.current_version,
        "model_version_name": model_manager.version_name,
        "pipeline_metrics": {
            "visual_ext_s": round(t1 - t0, 3),
            "audio_ext_s": round(t2 - t1, 3),
            "text_ext_s": round(t3 - t2, 3),
            "inference_s": round(t4 - t3, 3)
        },
        "modality_weights": [round(w, 4) for w in avg_attention],
        "temporal_probs": [round(p, 4) for p in temporal_probs]
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

class PipelineMetrics(BaseModel):
    visual_ext_s: float
    audio_ext_s: float
    text_ext_s: float
    inference_s: float

class PredictionResponse(BaseModel):
    video_name:         str
    prediction:         str
    confidence:         float
    prob_brainrot:      float
    prob_non_brainrot:  float
    transcript:         str
    folds_used:         int
    per_fold_probs:     List[float]
    processing_time_s:  float
    model_version:      str
    model_version_name: str
    pipeline_metrics:   PipelineMetrics
    modality_weights:   List[float]
    temporal_probs:     List[float]


class HealthResponse(BaseModel):
    status:          str
    device:          str
    folds_loaded:    List[int]
    models_ready:    bool
    model_version:   str
    model_version_name: str


class ModelInfoResponse(BaseModel):
    model_version:    str
    model_version_name: str
    checkpoint_dir:   str
    folds_available:  List[int]
    device:           str
    visual_dim:       int
    audio_dim:        int
    text_dim:         int
    fusion_dim:       int


class ModelVersionInfo(BaseModel):
    version_id:     str
    name:           str
    checkpoint_dir: str
    n_folds:        int
    is_active:      bool


class ModelsListResponse(BaseModel):
    active_version: str
    versions:       List[ModelVersionInfo]


class SwitchResponse(BaseModel):
    message:        str
    active_version: str
    folds_loaded:   List[int]


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
        status             = "ok",
        device             = str(DEVICE),
        folds_loaded       = model_manager.available_folds,
        models_ready       = model_manager._loaded,
        model_version      = model_manager.current_version or "",
        model_version_name = model_manager.version_name,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Status"])
async def model_info():
    """Get information about the loaded model configuration."""
    version_cfg = MODEL_VERSIONS.get(model_manager.current_version, {})
    return ModelInfoResponse(
        model_version      = model_manager.current_version or "",
        model_version_name = model_manager.version_name,
        checkpoint_dir     = version_cfg.get("checkpoint_dir", ""),
        folds_available    = model_manager.available_folds,
        device             = str(DEVICE),
        visual_dim         = VISUAL_FEAT_DIM,
        audio_dim          = AUDIO_FEAT_DIM,
        text_dim           = TEXT_FEAT_DIM,
        fusion_dim         = FUSION_DIM,
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Model Versions"])
async def list_models():
    """List all available model versions and which one is currently active."""
    versions = []
    for vid, cfg in MODEL_VERSIONS.items():
        versions.append(ModelVersionInfo(
            version_id     = vid,
            name           = cfg["name"],
            checkpoint_dir = cfg["checkpoint_dir"],
            n_folds        = cfg["n_folds"],
            is_active      = (vid == model_manager.current_version),
        ))
    return ModelsListResponse(
        active_version = model_manager.current_version or "",
        versions       = versions,
    )


@app.post("/models/switch", response_model=SwitchResponse, tags=["Model Versions"])
async def switch_model(version: str):
    """
    Switch the active model version.
    Available versions can be listed via GET /models.
    """
    if version not in MODEL_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown version: '{version}'. Available: {list(MODEL_VERSIONS.keys())}"
        )
    try:
        model_manager.switch_version(version)
        return SwitchResponse(
            message        = f"Switched to '{version}' ({MODEL_VERSIONS[version]['name']})",
            active_version = model_manager.current_version,
            folds_loaded   = model_manager.available_folds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import StreamingResponse

async def telemetry_generator(task_id: str):
    while True:
        if task_id not in ACTIVE_TASKS_STATE:
            yield "data: {\"status\": \"done\"}\n\n"
            break
        state = ACTIVE_TASKS_STATE[task_id]
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / (1024**3)
        gpu_percent = 0.0
        vram_gb = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                vram_gb = gpu.memoryUsed / 1024
        except:
            pass
            
        payload = {
            "cpu_percent": cpu,
            "ram_gb": round(ram, 1),
            "gpu_percent": round(gpu_percent, 1),
            "vram_gb": round(vram_gb, 1),
            "stage": state.get("stage", "Initializing"),
            "log_message": state.get("log", "Setting up telemetry...")
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.5)

@app.get("/telemetry/{task_id}", tags=["Telemetry"])
async def get_telemetry(task_id: str):
    return StreamingResponse(telemetry_generator(task_id), media_type="text/event-stream")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    video: UploadFile = File(..., description="Video file to classify (mp4, avi, mov)"),
    model_version: Optional[str] = None,
    task_id: Optional[str] = None,
):
    if task_id: ACTIVE_TASKS_STATE[task_id] = {"stage": "Uploading", "log": "Receiving video..."}
    """
    Classify a video using the best model from Fold 1.
    Optionally specify model_version ('default' or 'no_yt') as a query parameter.
    Faster than ensemble — good for quick classification.
    """
    if not model_manager._loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    if model_version and model_version not in MODEL_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model version: '{model_version}'. Available: {list(MODEL_VERSIONS.keys())}"
        )

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
        result     = await asyncio.to_thread(run_inference, tmp_path, [min(model_manager.available_folds)], model_version, task_id)
        elapsed    = time.time() - start_time

        return PredictionResponse(
            video_name        = video.filename,
            processing_time_s = round(elapsed, 2),
            **result,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if task_id and task_id in ACTIVE_TASKS_STATE:
            del ACTIVE_TASKS_STATE[task_id]


@app.post("/predict/ensemble", response_model=PredictionResponse, tags=["Inference"])
async def predict_ensemble(
    video: UploadFile = File(..., description="Video file to classify (mp4, avi, mov)"),
    model_version: Optional[str] = None,
    task_id: Optional[str] = None,
):
    if task_id: ACTIVE_TASKS_STATE[task_id] = {"stage": "Uploading", "log": "Receiving ensemble video..."}
    """
    Classify a video using ALL fold models and average their predictions.
    Optionally specify model_version ('default' or 'no_yt') as a query parameter.
    More accurate than single-fold — recommended for final results.
    """
    if not model_manager._loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    if model_version and model_version not in MODEL_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model version: '{model_version}'. Available: {list(MODEL_VERSIONS.keys())}"
        )

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
        result     = await asyncio.to_thread(run_inference, tmp_path, model_manager.available_folds, model_version, task_id)
        elapsed    = time.time() - start_time

        return PredictionResponse(
            video_name        = video.filename,
            processing_time_s = round(elapsed, 2),
            **result,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if task_id and task_id in ACTIVE_TASKS_STATE:
            del ACTIVE_TASKS_STATE[task_id]


# =============================================================================
# URL-BASED INFERENCE (Download from TikTok / Instagram / YouTube)
# =============================================================================

class URLRequest(BaseModel):
    url: str
    model_version: Optional[str] = None
    mode: Optional[str] = "ensemble"       # 'single' or 'ensemble'
    task_id: Optional[str] = None

class URLValidateResponse(BaseModel):
    valid: bool
    platform: Optional[str] = None
    url: str


@app.post("/validate/url", response_model=URLValidateResponse, tags=["URL Download"])
async def validate_url(payload: URLRequest):
    """Validate a URL and detect the platform (youtube, tiktok, instagram)."""
    url = payload.url.strip()
    if not url:
        return URLValidateResponse(valid=False, platform=None, url=url)

    url_lower = url.lower()
    supported_patterns = [
        'youtube.com', 'youtu.be',
        'tiktok.com',
        'instagram.com', 'instagr.am',
    ]
    is_valid = any(p in url_lower for p in supported_patterns)
    platform = detect_platform(url) if is_valid else None
    return URLValidateResponse(valid=is_valid, platform=platform, url=url)


# ── Video Preview (metadata extraction without downloading) ──────────────────

class PreviewResponse(BaseModel):
    title: Optional[str] = None
    thumbnail: Optional[str] = None
    duration: Optional[float] = None
    uploader: Optional[str] = None
    platform: Optional[str] = None
    success: bool = False


@app.post("/preview/url", response_model=PreviewResponse, tags=["URL Download"])
async def preview_url(payload: URLRequest):
    """
    Extract video metadata (title, thumbnail, duration) from a URL
    without downloading the full video. Powers the frontend preview card.
    """
    import yt_dlp
    url = payload.url.strip()
    if not url:
        return PreviewResponse(success=False)

    platform = detect_platform(url)

    def _extract():
        opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'ignoreerrors': True,
            'socket_timeout': 15,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return None
            return {
                'title': info.get('title'),
                'thumbnail': info.get('thumbnail'),
                'duration': info.get('duration'),
                'uploader': info.get('uploader') or info.get('channel'),
            }

    try:
        result = await asyncio.to_thread(_extract)
        if result is None:
            return PreviewResponse(success=False, platform=platform)

        # Proxy the thumbnail URL through our server to avoid CORS issues
        thumb_url = result.get('thumbnail')
        if thumb_url:
            # Encode and route through our proxy
            import urllib.parse
            proxied = f"/proxy/thumbnail?url={urllib.parse.quote(thumb_url, safe='')}"
            result['thumbnail'] = proxied

        return PreviewResponse(
            success=True,
            platform=platform,
            **result,
        )
    except Exception as e:
        print(f"[Preview] Error extracting metadata: {e}")
        return PreviewResponse(success=False, platform=platform)


@app.get("/proxy/thumbnail", tags=["URL Download"])
async def proxy_thumbnail(url: str):
    """
    Proxy a thumbnail image to avoid CORS issues.
    The frontend calls this with the encoded thumbnail URL.
    """
    import httpx
    from fastapi.responses import Response

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp.raise_for_status()
            content_type = resp.headers.get('content-type', 'image/jpeg')
            return Response(
                content=resp.content,
                media_type=content_type,
                headers={'Cache-Control': 'public, max-age=3600'}
            )
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to fetch thumbnail")


@app.post("/predict/url", response_model=PredictionResponse, tags=["URL Download"])
async def predict_from_url(payload: URLRequest):
    """
    Download a video from a URL (TikTok, Instagram, YouTube) and classify it.
    Uses the video_downloader engine under the hood.
    """
    if not model_manager._loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    url = payload.url.strip()
    model_version = payload.model_version
    mode = payload.mode or "ensemble"
    task_id = payload.task_id

    if task_id:
        ACTIVE_TASKS_STATE[task_id] = {"stage": "Validating URL", "log": f"Checking URL: {url}"}

    if model_version and model_version not in MODEL_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model version: '{model_version}'. Available: {list(MODEL_VERSIONS.keys())}"
        )

    # Detect platform
    platform = detect_platform(url)

    # Create a temporary directory for downloads
    tmp_dir = os.path.join(tempfile.gettempdir(), f"brainrot_dl_{uuid.uuid4().hex[:8]}")
    os.makedirs(tmp_dir, exist_ok=True)

    downloaded_path = None
    try:
        # Download the video
        if task_id:
            ACTIVE_TASKS_STATE[task_id] = {
                "stage": "Downloading Video",
                "log": f"[{platform.upper()}] Downloading from: {url}"
            }

        # Use the video_downloader module — run in thread to avoid blocking
        def _download():
            scraper = get_scraper(
                platform,
                max_duration_seconds=600,   # allow up to 10min videos
                download_archive=None,      # don't use archive for one-off analysis
            )
            # Disable archive for single analysis use
            scraper.ydl_opts.pop('download_archive', None)
            scraper.download_videos([url], output_dir=tmp_dir)

        await asyncio.to_thread(_download)

        # Find the downloaded file
        downloaded_files = [
            f for f in os.listdir(tmp_dir)
            if f.lower().endswith(('.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv'))
        ]

        if not downloaded_files:
            raise HTTPException(
                status_code=400,
                detail="Download failed — no video file was produced. The URL may be invalid, geo-restricted, or require authentication."
            )

        downloaded_path = os.path.join(tmp_dir, downloaded_files[0])
        video_display_name = f"[{platform.upper()}] {downloaded_files[0]}"

        if task_id:
            ACTIVE_TASKS_STATE[task_id] = {
                "stage": "Download Complete",
                "log": f"File: {downloaded_files[0]} — starting inference pipeline..."
            }

        # Determine fold nums
        if mode == 'ensemble':
            fold_nums = model_manager.available_folds
        else:
            fold_nums = [min(model_manager.available_folds)]

        start_time = time.time()
        result = await asyncio.to_thread(
            run_inference, downloaded_path, fold_nums, model_version, task_id
        )
        elapsed = time.time() - start_time

        return PredictionResponse(
            video_name=video_display_name,
            processing_time_s=round(elapsed, 2),
            **result,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if task_id and task_id in ACTIVE_TASKS_STATE:
            del ACTIVE_TASKS_STATE[task_id]
