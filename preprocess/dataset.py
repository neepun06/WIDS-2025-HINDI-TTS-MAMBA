# preprocess/dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocess.text import load_transcripts, CharTokenizer


class HindiTTSDataset(Dataset):
    def __init__(self, root="data/hindi", max_items=500):
        meta_path = os.path.join(root, "metadata", "transcripts.txt")
        self.mel_dir = os.path.join(root, "mels")

        pairs = load_transcripts(meta_path)[:max_items]
        texts = [t for _, t in pairs]
        self.tokenizer = CharTokenizer(texts)

        self.items = []
        for wav_id, text in pairs:
            mel_path = os.path.join(self.mel_dir, wav_id + ".npy")
            if os.path.exists(mel_path):
                self.items.append((mel_path, text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mel_path, text = self.items[idx]

        mel = np.load(mel_path)
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)  # normalize
        mel = torch.tensor(mel, dtype=torch.float)

        tokens = torch.tensor(
            self.tokenizer.encode(text),
            dtype=torch.long
        )

        return tokens, mel
