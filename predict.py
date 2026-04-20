"""
=============================================================================
BRAINROT DETECTOR — LOCAL INFERENCE SCRIPT
=============================================================================
Usage:
    python predict.py --video path/to/video.mp4
    python predict.py --video path/to/video.mp4 --model BEST_fold1.pt
    python predict.py --video path/to/video.mp4 --all_folds

Requirements:
    pip install torch torchvision opencv-python librosa transformers==4.40.0 h5py numpy
=============================================================================
"""

import os
import gc
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ── CONFIG — update paths to match your setup ────────────────────────────────
CHECKPOINT_DIR  = "./checkpoints"          # folder containing BEST_fold*.pt files
N_FOLDS         = 3                        # number of folds you trained

# ── MODEL VERSIONS ────────────────────────────────────────────────────────────
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
VISUAL_FEAT_DIM = 1280
AUDIO_FEAT_DIM  = 2048
TEXT_FEAT_DIM   = 768
FUSION_DIM      = 256
NUM_CLASSES     = 2
MAX_TEXT_LEN    = 128
N_MELS          = 128
AUDIO_MAX_FRAMES= 300
IMG_SIZE        = 224
MAX_FRAMES      = 16                       # frames to sample from video

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms matching EfficientNet V1 ImageNet training
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
# FEATURE EXTRACTION FROM VIDEO FILE
# =============================================================================

def extract_visual_features(video_path: str) -> np.ndarray:
    """Extract EfficientNet-B0 visual features from video frames."""
    import cv2
    
    print("  [Visual] Extracting frames & features...")
    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
         print("  [!] Could not open video.")
         return np.zeros((1, VISUAL_FEAT_DIM), dtype=np.float32)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Sample evenly spaced frames
    indices = np.linspace(0, max(total - 1, 0), MAX_FRAMES, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixel_values = img_transform(frame)
            frames.append(pixel_values)
    cap.release()

    if not frames:
        print("  [!] Could not read frames from video.")
        return np.zeros((1, VISUAL_FEAT_DIM), dtype=np.float32)

    # Use torchvision EfficientNet (weights will auto-download on first run)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(DEVICE)
    model.classifier = nn.Identity()
    model.eval()

    batch = torch.stack(frames).to(DEVICE)
    with torch.no_grad():
        feats = model(batch)
        mean_feat = feats.mean(dim=0, keepdim=True)
    
    res = mean_feat.cpu().numpy()
    
    del model
    gc.collect()

    print(f"  [Visual] Done. Shape: {res.shape}")
    return res.astype(np.float32)


def extract_audio_features(video_path: str) -> np.ndarray:
    """Extract mel spectrogram features from video audio track."""
    try:
        import librosa
    except ImportError:
        print("  [!] Missing: pip install librosa")
        sys.exit(1)

    print("  [Audio] Extracting audio...")
    audio_path = video_path.replace(os.path.splitext(video_path)[1], "_tmp_audio.wav")

    # Extract audio using ffmpeg
    ret = os.system(f'ffmpeg -i "{video_path}" -ac 1 -ar 22050 "{audio_path}" -y -loglevel quiet')
    if ret != 0 or not os.path.exists(audio_path):
        print("  [Audio] ffmpeg failed or no audio track. Using zeros.")
        return np.zeros((1, AUDIO_FEAT_DIM), dtype=np.float32)

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        if os.path.exists(audio_path): os.remove(audio_path)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel_t = mel_db.T  # (time, 128)

        if mel_t.shape[0] >= AUDIO_MAX_FRAMES:
            mel_t = mel_t[:AUDIO_MAX_FRAMES, :]
        else:
            pad = np.zeros((AUDIO_MAX_FRAMES - mel_t.shape[0], N_MELS), dtype=np.float32)
            mel_t = np.vstack([mel_t, pad])

        flat = mel_t.flatten()[None]  # (1, 38400)
        
        if flat.shape[1] >= AUDIO_FEAT_DIM:
            feat = flat[:, :AUDIO_FEAT_DIM]
        else:
            feat = np.zeros((1, AUDIO_FEAT_DIM), dtype=np.float32)
            feat[:, :flat.shape[1]] = flat

        print(f"  [Audio] Done. Shape: {feat.shape}")
        return feat.astype(np.float32)
    except Exception as e:
        print(f"  [!] Audio extraction failed: {e}")
        if os.path.exists(audio_path): os.remove(audio_path)
        return np.zeros((1, AUDIO_FEAT_DIM), dtype=np.float32)


def extract_text_features(video_path: str) -> np.ndarray:
    """Extract DistilBERT CLS token features from video audio transcript."""
    from transformers import DistilBertModel, DistilBertTokenizer

    print("  [Text] Transcribing audio...")
    transcript = ""
    try:
        import whisper
        audio_path = video_path.replace(os.path.splitext(video_path)[1], "_tmp_whisper.wav")
        os.system(f'ffmpeg -i "{video_path}" -ac 1 -ar 16000 "{audio_path}" -y -loglevel quiet')
        if os.path.exists(audio_path):
            whisper_model = whisper.load_model("tiny")
            result        = whisper_model.transcribe(audio_path)
            transcript    = result.get("text", "").strip()
            os.remove(audio_path)
            del whisper_model
    except Exception:
        pass

    if not transcript:
        transcript = "no speech detected"

    print(f"  [Text] Transcript: \"{transcript[:50]}...\"")
    
    tokenizer  = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE)
    bert_model.eval()

    encoded = tokenizer(
        [transcript],
        padding='max_length',
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors='pt'
    ).to(DEVICE)

    with torch.no_grad():
        output = bert_model(**encoded)
    
    feat = output.last_hidden_state[:, 0, :].cpu().numpy()
    
    del bert_model, tokenizer
    gc.collect()

    print(f"  [Text] Done. Shape: {feat.shape}")
    return feat.astype(np.float32)


# =============================================================================
# INFERENCE
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> BrainrotModel:
    model = BrainrotModel().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_single(video_path: str, checkpoint_path: str, device: torch.device) -> dict:
    """Run inference on a single video file."""

    print(f"\n{'─'*50}")
    print(f"📹 Video   : {os.path.basename(video_path)}")
    print(f"🧠 Model   : {os.path.basename(checkpoint_path)}")
    print(f"📦 Version : {getattr(predict_single, '_version_name', 'N/A')}")
    print(f"{'─'*50}")

    # Extract features
    vis_feat = extract_visual_features(video_path)
    aud_feat = extract_audio_features(video_path)
    txt_feat = extract_text_features(video_path)

    # Convert to tensors
    vis_t = torch.tensor(vis_feat, dtype=torch.float32).to(device)
    aud_t = torch.tensor(aud_feat, dtype=torch.float32).to(device)
    txt_t = torch.tensor(txt_feat, dtype=torch.float32).to(device)

    # Load model and run inference
    model  = load_model(checkpoint_path, device)
    with torch.no_grad():
        logits = model(vis_t, aud_t, txt_t)
        probs  = torch.softmax(logits, dim=-1)

    prob_brainrot     = float(probs[0][1].cpu())
    prob_non_brainrot = float(probs[0][0].cpu())
    predicted_class   = "BRAINROT" if prob_brainrot >= 0.5 else "NON-BRAINROT"

    return {
        "video":           os.path.basename(video_path),
        "prediction":      predicted_class,
        "confidence":      max(prob_brainrot, prob_non_brainrot),
        "prob_brainrot":   prob_brainrot,
        "prob_non_brainrot": prob_non_brainrot,
    }


def predict_ensemble(video_path: str, device: torch.device) -> dict:
    """
    Run inference using all available fold models and average predictions.
    More robust than using a single fold.
    """
    all_probs = []

    for fold_num in range(1, N_FOLDS + 1):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"BEST_fold{fold_num}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [!] Checkpoint not found: {ckpt_path} — skipping fold {fold_num}")
            continue

        print(f"\n  Running Fold {fold_num} model...")
        result = predict_single(video_path, ckpt_path, device)
        all_probs.append(result["prob_brainrot"])
        print(f"  Fold {fold_num} → Brainrot prob: {result['prob_brainrot']:.4f}")

    if not all_probs:
        print("[!] No checkpoints found. Check CHECKPOINT_DIR path.")
        sys.exit(1)

    avg_prob          = float(np.mean(all_probs))
    predicted_class   = "BRAINROT" if avg_prob >= 0.5 else "NON-BRAINROT"
    confidence        = avg_prob if predicted_class == "BRAINROT" else 1 - avg_prob

    return {
        "video":             os.path.basename(video_path),
        "prediction":        predicted_class,
        "confidence":        confidence,
        "prob_brainrot":     avg_prob,
        "prob_non_brainrot": 1 - avg_prob,
        "folds_used":        len(all_probs),
        "per_fold_probs":    all_probs,
    }


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_result(result: dict):
    label     = result["prediction"]
    conf      = result["confidence"] * 100
    prob_br   = result["prob_brainrot"] * 100
    prob_nbr  = result["prob_non_brainrot"] * 100

    emoji = "🧠💀" if label == "BRAINROT" else "✅"

    print(f"\n{'='*50}")
    print(f"  RESULT: {emoji}  {label}")
    print(f"{'='*50}")
    print(f"  Confidence      : {conf:.1f}%")
    print(f"  P(Brainrot)     : {prob_br:.1f}%")
    print(f"  P(Non-Brainrot) : {prob_nbr:.1f}%")

    if "folds_used" in result:
        print(f"  Folds used      : {result['folds_used']}")
        per_fold = [f"{p*100:.1f}%" for p in result["per_fold_probs"]]
        print(f"  Per-fold probs  : {', '.join(per_fold)}")

    # Visual confidence bar
    bar_len  = 30
    filled   = int(bar_len * result["prob_brainrot"])
    bar      = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  Non-Brainrot [{bar}] Brainrot")
    print(f"{'='*50}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global CHECKPOINT_DIR, N_FOLDS

    # Build help text showing available versions
    version_help = "Model version to use. Available: " + ", ".join(
        f"'{k}' ({v['name']})" for k, v in MODEL_VERSIONS.items()
    ) + f" (default: {DEFAULT_VERSION})"

    parser = argparse.ArgumentParser(
        description="Brainrot Detector — classify a video as brainrot or not"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file (mp4, avi, mov, etc.)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to a specific .pt checkpoint file (optional)"
    )
    parser.add_argument(
        "--all_folds", action="store_true",
        help="Use all fold checkpoints and average predictions (ensemble)"
    )
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Directory containing BEST_fold*.pt files (overrides --model_version)"
    )
    parser.add_argument(
        "--model_version", default=DEFAULT_VERSION,
        choices=list(MODEL_VERSIONS.keys()),
        help=version_help,
    )
    args = parser.parse_args()

    # Validate video path
    if not os.path.exists(args.video):
        print(f"[!] Video file not found: {args.video}")
        sys.exit(1)

    # Resolve checkpoint dir from model version (or explicit override)
    if args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
        version_name   = "Custom"
    else:
        version_cfg    = MODEL_VERSIONS[args.model_version]
        CHECKPOINT_DIR = version_cfg["checkpoint_dir"]
        N_FOLDS        = version_cfg["n_folds"]
        version_name   = version_cfg["name"]

    # Store version name for display in predict_single
    predict_single._version_name = f"{args.model_version} ({version_name})"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device  : {device}")
    print(f"📦 Version : {args.model_version} ({version_name})")
    print(f"📂 Checkpts: {CHECKPOINT_DIR}")

    # Run inference
    if args.all_folds:
        print("\n🔁 Running ensemble inference across all folds...")
        result = predict_ensemble(args.video, device)
    elif args.model:
        if not os.path.exists(args.model):
            print(f"[!] Model file not found: {args.model}")
            sys.exit(1)
        result = predict_single(args.video, args.model, device)
    else:
        # Default: use BEST_fold1.pt
        default_ckpt = os.path.join(CHECKPOINT_DIR, "BEST_fold1.pt")
        if not os.path.exists(default_ckpt):
            print(f"[!] Default checkpoint not found: {default_ckpt}")
            print(f"    Use --model or --all_folds flags.")
            sys.exit(1)
        result = predict_single(args.video, default_ckpt, device)

    print_result(result)


if __name__ == "__main__":
    main()
