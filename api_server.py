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
import torchvision.models as models
import torchvision.transforms as transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

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

# ── MODEL VERSIONS ──────────────────────────────────────────────────────────
MODEL_VERSIONS = {
    "default": {
        "name": "Default (Full Dataset)",
        "checkpoint_dir": "./checkpoints",
        "n_folds": 3,
    },
    "no_yt": {
        "name": "No YouTube",
        "checkpoint_dir": "./checkpoint2",
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

        print("  [✓] Loading EfficientNet-B0 (PyTorch)...")
        self.eff_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(DEVICE)
        self.eff_model.classifier = nn.Identity()  # Remove top layer to get features
        self.eff_model.eval()

        print("  [✓] Loading DistilBERT (PyTorch)...")
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
                print(f"  [✓] Loaded BEST_fold{fold_num}.pt")
            else:
                print(f"  [!] Not found: {ckpt_path}")

        if not self.models:
            raise RuntimeError(f"No checkpoints found in: {checkpoint_dir}")

        self.current_version = version_id
        print(f"[ModelManager] Version '{version_id}' ready. Folds available: {list(self.models.keys())}")

    def load_all(self, version_id: str = None):
        """Initial load: feature extractors + default version fold models."""
        if version_id is None:
            version_id = DEFAULT_VERSION

        if not self._loaded:
            print(f"\n[ModelManager] Loading models on {DEVICE}...")
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixel_values = model_manager.img_transform(frame)
            frames.append(pixel_values)
    cap.release()

    if not frames:
        return np.zeros((1, VISUAL_FEAT_DIM), dtype=np.float32)

    # Average features across sampled frames
    batch = torch.stack(frames).to(DEVICE)
    with torch.no_grad():
        feats = model_manager.eff_model(batch) # (N, 1280)
        mean_feat = feats.mean(dim=0, keepdim=True) # (1, 1280)
    
    return mean_feat.cpu().numpy().astype(np.float32)


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
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        output = model_manager.bert_model(input_ids=input_ids, attention_mask=attention_mask)
    
    feat = output.last_hidden_state[:, 0, :].cpu().numpy()
    return feat.astype(np.float32), transcript


def run_inference(video_path: str, fold_nums: Optional[List[int]] = None, model_version: Optional[str] = None) -> dict:
    """Core inference logic — used by both single and ensemble endpoints."""
    # Auto-swap model version if requested
    if model_version:
        model_manager.ensure_version(model_version)

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
        "model_version":     model_manager.current_version,
        "model_version_name": model_manager.version_name,
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


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    video: UploadFile = File(..., description="Video file to classify (mp4, avi, mov)"),
    model_version: Optional[str] = None,
):
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
        result     = run_inference(tmp_path, fold_nums=[min(model_manager.available_folds)], model_version=model_version)
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


@app.post("/predict/ensemble", response_model=PredictionResponse, tags=["Inference"])
async def predict_ensemble(
    video: UploadFile = File(..., description="Video file to classify (mp4, avi, mov)"),
    model_version: Optional[str] = None,
):
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
        result     = run_inference(tmp_path, fold_nums=model_manager.available_folds, model_version=model_version)
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
