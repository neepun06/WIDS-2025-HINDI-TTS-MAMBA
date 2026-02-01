# preprocess/audio.py

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80


def wav_to_mel(wav_path):
    wav, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=1.0
    )

    mel = librosa.amplitude_to_db(mel, ref=np.max)
    return mel


def main():
    wav_dir = "data/hindi/wavs"
    out_dir = "data/hindi/mels"

    os.makedirs(out_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

    if len(wav_files) == 0:
        raise RuntimeError("No wav files found in data/hindi/wavs")

    for fname in wav_files:
        wav_path = os.path.join(wav_dir, fname)
        mel = wav_to_mel(wav_path)

        out_path = os.path.join(out_dir, fname.replace(".wav", ".npy"))
        np.save(out_path, mel)

        print(f"Saved mel: {out_path} | shape = {mel.shape}")

    # Plot first example for sanity
    mel = np.load(os.path.join(out_dir, wav_files[0].replace(".wav", ".npy")))
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, origin="lower", aspect="auto")
    plt.title("Mel Spectrogram")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
