# vocoder.py

import numpy as np
import librosa
import soundfile as sf

mel = np.load("output_mel.npy").T  # (80, T)

sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024

mel = mel.astype(np.float32)
mel = np.clip(mel, -4, 4)

wav = librosa.feature.inverse.mel_to_audio(
    mel,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_iter=64
)

sf.write("output.wav", wav, sr)
print("Saved output.wav")
