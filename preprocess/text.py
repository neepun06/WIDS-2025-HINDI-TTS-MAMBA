# preprocess/text.py

import re

def normalize(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_transcripts(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r'\(\s*(\S+)\s+"(.+?)"\s*\)', line)
            if m:
                wav_id, text = m.group(1), normalize(m.group(2))
                pairs.append((wav_id, text))
    return pairs


class CharTokenizer:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.pad = "<pad>"
        self.sos = "<s>"
        self.eos = "</s>"

        self.chars = [self.pad, self.sos, self.eos] + chars
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text):
        return (
            [self.stoi[self.sos]]
            + [self.stoi[c] for c in text if c in self.stoi]
            + [self.stoi[self.eos]]
        )

    def __len__(self):
        return len(self.chars)
