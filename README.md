# Hindi Text-to-Speech using Transformer and MAMBA Architectures

## Overview

This project implements a **Hindi Text-to-Speech (TTS)** system and presents a comparative study between a **traditional Transformer-based architecture** and a **MAMBA-inspired State Space Model (SSM)** architecture. The system converts input Hindi text into speech waveforms using mel-spectrogram prediction followed by waveform reconstruction.

The primary goal of this project is to explore whether **MAMBA-style sequence models**, which avoid explicit self-attention, can serve as an effective alternative to Transformers for TTS tasks.

## Dataset

- **Source:** IIT Madras Hindi Speech Dataset
- **Language:** Hindi only
- **Number of utterances used:** ~536
- **Audio format:** WAV files
- **Transcripts:** Provided via metadata files

No manual data cleaning or filtering was performed beyond what is implemented in the preprocessing scripts.

## Preprocessing Pipeline

### Audio Processing

- All audio files are **resampled to 22,050 Hz** using Librosa.
- Mel-spectrograms are extracted with:
  - n_fft = 1024
  - hop_length = 256
  - n_mels = 80

- Log-amplitude (dB) scaling is applied.
- Per-utterance **mean–variance normalization** is used.
- Mel-spectrograms are stored as .npy files for efficient loading.

### Text Processing

- Basic text normalization (whitespace cleanup).
- **Character-level tokenization** (no phonemes).
- Special tokens:
  - for padding
  - start-of-sequence
  - end-of-sequence

## Model Architectures

### 1\. Transformer-based TTS

- Token embedding + sinusoidal positional encoding
- Multi-layer Transformer encoder
- Linear projection to 80-dimensional mel-spectrograms
- Serves as the **baseline attention-based model**

### 2\. MAMBA-based TTS

- Attention-free architecture inspired by **state space sequence models**
- Consists of stacked MAMBA-style blocks with:
  - Gated linear projections
  - Depthwise temporal convolutions
  - Residual connections

- Designed to model long sequences efficiently

Both models predict mel-spectrograms directly from tokenized text.

## Training Setup

- **Loss function:** L1 loss between predicted and ground-truth mel-spectrograms
- **Optimizer:** Adam
- **Learning rate:** 3e-4
- **Epochs:** 120
- **Batch size:** 2
- **Gradient clipping:** Enabled
- **Checkpointing:**
  - Best model (lowest validation loss)
  - Last model per epoch

Training automatically uses **GPU if available**, otherwise falls back to CPU.

## Inference Pipeline

1.  Input Hindi text is tokenized.
2.  The trained TTS model predicts a mel-spectrogram.
3.  The time axis is stretched for improved audibility.
4.  The mel-spectrogram is saved as output_mel.npy.

## Vocoder

- Waveform reconstruction is performed using **Griffin–Lim** via Librosa.
- Uses inverse mel-spectrogram transformation.
- Output audio is saved as output.wav.

The same vocoder is used for both Transformer and MAMBA models to ensure a fair comparison.

## Repository Structure

```text
.
├── checkpoints_mamba
│   ├── best_model.pt
│   └── last_model.pt
├── checkpoints_transformer
│   ├── best_model.pt
│   └── last_model.pt
├── data
│   └── hindi
│       ├── wavs
│       ├── mels
│       └── metadata
├── models
│   ├── transformer_tts.py
│   └── mamba_tts.py
├── preprocess
│   ├── audio.py
│   ├── text.py
│   ├── dataset.py
│   └── collate.py
├── train.py
├── inference.py
├── vocoder.py
├── output_mel.npy
└── output.wav
```

---

## Key Contributions

- End-to-end Hindi TTS pipeline
- Side-by-side comparison of Transformer and MAMBA architectures
- Demonstration of attention-free sequence modeling for speech synthesis
- Clean, modular, and reproducible implementation

## Limitations and Future Work

- No objective MOS or human evaluation was conducted.
- Griffin–Lim vocoder limits audio quality.
- Future work could integrate neural vocoders (e.g., HiFi-GAN) and phoneme-based tokenization.

## Dataset Setup

The IIT Madras Hindi Speech Dataset is **not included** in this repository due to size and licensing constraints.

To use this project:

1. Download the dataset from the official IIT Madras website.
2. Organize the files in the following structure:

```text
data/hindi/
├── wavs/
└── metadata/
```

3. Generate mel-spectrograms by running:

```python
python preprocess/audio.py
```

## Environment Setup

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Python 3.10 or higher is recommended.
