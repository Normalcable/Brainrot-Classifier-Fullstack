# Multimodal Brainrot Detector: Improvement & Implementation Plan

This document outlines the pipeline for upgrading the current Brainrot Detector model to achieve higher accuracy and better performance, specifically optimized for **Google Colab Free Tier** and **Thesis Defense** requirements.

---

## 🚀 Phase 1: Feature Caching (Efficiency Pipeline)
To bypass the time limits and hardware constraints of Colab Free, **do not train end-to-end**. Instead, pre-extract features.

1.  **Extract & Save:** Run the dataset once through the backbone models.
    *   **Visual:** Save 16 frame vectors per video.
    *   **Audio:** Save the raw Mel Spectrogram.
    *   **Text:** Save the Whisper transcripts.
2.  **Format:** Save as `.npy` or `.h5` files directly to Google Drive.
3.  **Benefit:** Training the classifier will take **seconds per epoch**, allowing for rapid iteration and testing.

---

## 🧠 Phase 2: Enhanced Feature Engineering
Improve how the model "understands" each modality without increasing model size.

### 1. Visual: "Chaos-Aware" Pooling
*   **Current:** Mean-pooling only (loses temporal energy).
*   **Improvement:** Use **Concatenated Mean-Max Pooling**.
*   **Logic:** `Features = torch.cat([x.mean(dim=1), x.max(dim=1)], dim=-1)`
*   **Impact:** Captures both the average visual "vibe" and the sudden "peaks" of chaotic edits.

### 2. Audio: Statistical Signatures
*   **Current:** Flattening raw spectrogram pixels (fragile).
*   **Improvement:** Calculate **Global Statistics** (Mean, Std, Max) across the time axis for each frequency bin.
*   **Impact:** Detects audio distortion, high-frequency "ear-rape" memes, and rhythmic chaos more reliably than raw pixels.

### 3. Text: Domain-Specific Lexicon
*   **Improvement:** Add a manual **Slang Dictionary** feature.
*   **Logic:** Count occurrences of words like *Skibidi, Rizz, Gyatt, Sigma, Fanum* in the transcript.
*   **Impact:** Provides a direct "shortcut" for the model to identify Gen-Alpha brainrot slang that standard BERT might not fully weight.

---

## 🛠️ Phase 3: Architecture & Fusion
Keep the "Brain" of the model modern and research-ready.

1.  **Modality Projectors:** Simple Linear layers + LayerNorm to map all inputs to 256-dim.
2.  **Attention Fusion:** Use the existing learned attention mechanism to weight which modality (Visual, Audio, or Text) is most reliable for a specific video.
3.  **Classifier:** 2-layer MLP with `GELU` activation and `Dropout(0.5)` to prevent overfitting on small datasets.

---

## 📊 Phase 4: Thesis Validation Strategy
Use these steps to make your defense more scientifically robust:

1.  **K-Fold Cross-Validation:** Train on 3 or 5 folds to prove your results aren't accidental.
2.  **Ablation Study:** Show what happens when you remove one modality (e.g., "Model without Audio"). This proves your model is truly multimodal.
3.  **Inference Speed:** Benchmark your new pipeline. Since you are using a lightweight MLP for final inference, it will be extremely fast for real-world deployment.

---

## 🏁 Summary Checklist
- [ ] Pre-extract features and save to `.npy`.
- [ ] Update Visual pooling to `Mean + Max`.
- [ ] Update Audio features to `Statistical (Mean/Std/Max)`.
- [ ] Add `Keyword Count` to the Text modality.
- [ ] Train the Fusion Classifier on the cached features.
