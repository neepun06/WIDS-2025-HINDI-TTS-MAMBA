# preprocess/collate.py

import torch


def collate_fn(batch):
    tokens, mels = zip(*batch)

    max_t = max(t.size(0) for t in tokens)
    max_m = max(m.size(1) for m in mels)

    token_pad = torch.zeros(len(tokens), max_t, dtype=torch.long)
    mel_pad = torch.zeros(len(mels), 80, max_m)

    for i, (t, m) in enumerate(zip(tokens, mels)):
        token_pad[i, :t.size(0)] = t
        mel_pad[i, :, :m.size(1)] = m

    return token_pad, mel_pad
