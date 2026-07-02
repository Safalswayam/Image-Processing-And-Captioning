# Image Captioning: CNN+LSTM vs. ViT+GPT-2

A dual-pipeline deep learning project that generates natural-language captions for images, implementing and benchmarking two fundamentally different architectures side by side:

1. **CNN + LSTM** — DenseNet201 feature extraction with an LSTM-based sequence decoder (TensorFlow/Keras)
2. **ViT + GPT-2** — Vision Transformer encoder fused with a GPT-2 language model decoder via cross-attention (PyTorch/HuggingFace)

The project covers the full pipeline end to end: data preprocessing and augmentation, feature extraction, model training, BLEU-score evaluation, visualization, inference, and performance benchmarking.

---

## Table of Contents

- [Overview](#overview)
- [Architectures](#architectures)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Known Issues](#known-issues)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)

---

## Overview

Given an input image, both models produce a descriptive caption (e.g. *"a dog running through a field of grass"*). The notebook trains both models on the same dataset and split, then compares them on:

- **BLEU-1 / BLEU-2 / BLEU-4** scores
- **Inference latency** (average, min, max, std. dev. across repeated runs)
- **Qualitative caption quality** on sample images

This side-by-side setup makes it easy to reason about the classic CNN-RNN approach vs. a modern transformer-based approach for the same task.

## Architectures

### 1. CNN + LSTM (`create_cnn_lstm_model`)

| Component | Detail |
|---|---|
| Image encoder | DenseNet201 (ImageNet-pretrained), penultimate layer → 1920-d feature vector |
| Image projection | `Dense(256, relu)` → reshaped to `(1, 256)` |
| Text encoder | `Embedding(vocab_size, 256)` over the caption sequence |
| Fusion | Image + text embeddings concatenated, passed through `LSTM(256)` |
| Head | `Dropout → Add(residual) → Dense(128, relu) → Dropout → Dense(vocab_size, softmax)` |
| Training | Teacher forcing via a custom Keras `Sequence` generator (`CustomDataGenerator`), one training pair per partial sequence |
| Callbacks | `ModelCheckpoint`, `EarlyStopping(patience=5)`, `ReduceLROnPlateau` |

### 2. ViT + GPT-2 (`VisionTransformerCaptioning`)

| Component | Detail |
|---|---|
| Image encoder | `google/vit-base-patch16-224-in21k` (HuggingFace) |
| Text decoder | `gpt2` (HuggingFace `GPT2LMHeadModel`) |
| Fusion | Linear projection of ViT hidden states into GPT-2 embedding space + multi-head cross-attention (8 heads) |
| Tokens | Custom `<|startoftext|>` / `<|endoftext|>` markers wrapping each caption |
| Training | `ViTCaptioningTrainer` class — differential learning rates (decoder fine-tuned at 0.1× the head/projection LR), gradient clipping, `StepLR` scheduler |
| Data loading | `Flickr8kDataset` (see `vit_data_loader.py`) — falls back to a black placeholder image if a file is missing |

## Project Structure

```
Img_To_Caption/
├── image_captioning_notebook.ipynb   # End-to-end pipeline (20 sections / 48 cells)
├── vit_data_loader.py                # Standalone PyTorch Dataset + DataLoader for the ViT+GPT-2 pipeline
├── cnn_lstm_final.h5                 # Trained CNN+LSTM model weights (Keras/TensorFlow)
└── README.md
```

The notebook is organized into 20 clearly labeled sections:

1. Setup & Imports
2. Data Loading & Caption Preprocessing
3. Image Augmentation Functions
4. Train/Validation Split
5. Feature Extraction (DenseNet201) for CNN+LSTM
6. Tokenization & Sequences for CNN+LSTM
7. Data Generator for CNN+LSTM Training
8. CNN+LSTM Model Definition
9. ViT+GPT Dataset and Model Classes
10. Training Loop — CNN+LSTM
11. Training Loop — ViT+GPT
12. Visualizations (training curves, sample predictions, architecture comparison, dataset stats)
13. BLEU Score Implementation
14. BLEU Score Visualization & Model Comparison
15. Advanced Caption Generation & Comparison
16. Model Performance Analysis & Insights
17. Model Saving & Loading
18. Production-Ready Inference Class
19. Performance Benchmarking (speed analysis)
20. Final Summary & Next Steps

## Prerequisites

- Python 3.9+
- A CUDA-capable GPU is strongly recommended (ViT+GPT-2 training and DenseNet201 feature extraction are compute-heavy on CPU)
- ~10+ GB free disk space (pretrained weights + dataset + augmented images + checkpoints)

## Installation

```bash
git clone <repository-url>
cd Img_To_Caption

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Since no `requirements.txt` is bundled yet, install the core dependencies directly:

```bash
pip install tensorflow torch torchvision transformers \
            numpy pandas pillow matplotlib seaborn \
            tqdm nltk
```

Then, from Python, fetch the NLTK tokenizer data used for BLEU scoring:

```python
import nltk
nltk.download("punkt")
```

## Dataset

This project is built around the **Flickr8k** dataset (8,000 images, 5 captions each), but the raw data is **not included** in this repository/archive. Before running the notebook, download the dataset and arrange it as:

```
Img_To_Caption/
├── captions.txt      # columns: image, caption
└── Images/            # all Flickr8k .jpg files
```

The `captions.txt` file is expected in the format:

```
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs.
...
```

> Any dataset following this same `image,caption` CSV schema will work — the pipeline is not hard-coded to Flickr8k specifics beyond this format.

### Data augmentation

`create_augmented_data()` (Section 3) optionally expands the dataset via four deterministic transforms per image — 45° rotation, 90° rotation, horizontal flip, vertical flip — writing results to `Augmented_Images/` and `augmented_captions.txt`. This step is disabled by default in the notebook (commented out) since it's time-consuming; uncomment it to use the augmented set for training.

## Usage

1. Place `captions.txt` and `Images/` alongside the notebook (see [Dataset](#dataset)).
2. Launch Jupyter and open `image_captioning_notebook.ipynb`:
   ```bash
   jupyter notebook image_captioning_notebook.ipynb
   ```
3. Run cells sequentially from top to bottom. Key stages:
   - **Sections 1–4**: load, clean, and split the data
   - **Sections 5–8**: extract DenseNet201 features and build the CNN+LSTM model
   - **Section 9**: define the ViT+GPT-2 architecture
   - **Sections 10–11**: train both models (epochs are reduced for demo purposes — increase for production-quality results)
   - **Sections 12–16**: visualize training curves, generate BLEU scores, and compare models
   - **Sections 17–19**: persist trained artifacts and benchmark inference speed

### Using the trained CNN+LSTM model directly

A pretrained checkpoint, `cnn_lstm_final.h5`, is included in this repository. To load it standalone:

```python
import tensorflow as tf

model = tf.keras.models.load_model("cnn_lstm_final.h5")
```

Note that you'll still need the fitted `tokenizer_cnn` (Keras `Tokenizer`) and the DenseNet201 `feature_extractor` used at training time to preprocess new images and decode predictions — these are produced in Sections 5–6 of the notebook and are not bundled as separate artifacts. Re-run those cells, or persist the tokenizer yourself (via `pickle`, as shown in Section 17's `save_models_and_components()`) for repeatable inference.

### Standalone ViT+GPT-2 data loading

`vit_data_loader.py` can be imported independently of the notebook:

```python
from vit_data_loader import create_vit_loaders

train_loader, val_loader = create_vit_loaders(
    train_data, val_data, IMAGE_PATH="Images",
    tokenizer_vit=tokenizer_vit,
    feature_extractor=feature_extractor,
    batch_size=8,
)
```

## Evaluation

Both models are scored with **BLEU-1, BLEU-2, and BLEU-4** (`calculate_bleu_scores`, Section 13), using NLTK's `sentence_bleu` with smoothing (`SmoothingFunction().method4`) to avoid zero scores on short captions. Special tokens (`startseq`, `endseq`, `<|startoftext|>`, `<|endoftext|>`) are stripped before scoring.

`plot_bleu_comparison()` (Section 14) visualizes BLEU scores side by side, and `analyze_model_performance()` (Section 16) summarizes strengths/weaknesses of each architecture.

## Results

The notebook's final summary (`generate_final_summary()`, Section 20) reports BLEU-1/2/4 for both models and recommends a primary model based on aggregate BLEU performance, with the other retained as a backup/comparison model. Exact scores depend on your training run (epochs, dataset size, augmentation) — re-run Sections 10–16 to reproduce metrics for your own setup.

Inference speed is benchmarked in Section 19 (`benchmark_inference_speed`), running multiple timed passes per model and reporting mean, standard deviation, min, and max latency, along with a box plot and bar chart comparison.

## Known Issues

This project has undergone a code review, which surfaced several issues worth being aware of before relying on it in production:

- The BLEU evaluation function in Section 13 (`evaluate_model_bleu`) uses placeholder `"generated caption"` / `"reference caption"` strings rather than actual decoded predictions in one code path — the corrected version (with real caption decoding) appears later in the same section.
- `save_models_and_components()` references `best_vit_captioning_model.pth`, but the ViT training loop (Section 11) saves per-epoch checkpoints as `vit_epoch_{N}.pth` — there's no step that writes a file with that exact "best" filename, so loading will fail unless you rename/select the appropriate epoch checkpoint yourself.
- Several cells depend on variables defined only if earlier cells ran successfully and produced non-empty results (e.g. `cnn_lstm_results`, `vit_gpt_results`, `history_cnn`) — running cells out of order, or with a very small/incomplete dataset, can raise `NameError`s downstream.
- Image augmentation (Section 3) is disabled by default; results described elsewhere in the notebook that assume an augmented dataset won't match unless you explicitly enable it.

## Roadmap

- [ ] Add a `requirements.txt` / `environment.yml` for reproducible installs
- [ ] Extract the CNN+LSTM training/inference code into standalone `.py` modules (mirroring `vit_data_loader.py`)
- [ ] Fix the BLEU evaluation placeholder path in Section 13
- [ ] Align ViT+GPT-2 checkpoint naming between the training loop and `save_models_and_components()`
- [ ] Add a lightweight CLI or Flask/FastAPI wrapper around `ImageCaptioningInference` for serving
- [ ] Publish reproducible BLEU/latency benchmarks on the full (non-demo) training run

## Tech Stack

**Deep Learning:** TensorFlow / Keras, PyTorch, HuggingFace Transformers (ViT, GPT-2)
**Data & Evaluation:** NumPy, Pandas, NLTK (BLEU scoring)
**Imaging:** Pillow, torchvision
**Visualization:** Matplotlib, Seaborn
**Utilities:** tqdm

---

*Contributions and issue reports are welcome — see [Known Issues](#known-issues) above for a good starting point.*
