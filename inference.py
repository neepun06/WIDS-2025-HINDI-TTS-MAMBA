# inference.py

import torch
import numpy as np
from models.mamba_tts import MambaTTS
from preprocess.text import CharTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
texts = []
with open("data/hindi/metadata/transcripts.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.split('"')
        if len(parts) > 1:
            texts.append(parts[1])

tokenizer = CharTokenizer(texts)

# Load model
model = MambaTTS(vocab_size=len(tokenizer)).to(device)
ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Long inference text (important!)
text = (
    "राम जंगल में गए और वहाँ उन्होंने ऋषि से भेंट की "
    "और ज्ञान प्राप्त किया और अपने जीवन का मार्ग समझा"
)

tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)

with torch.no_grad():
    mel = model(tokens)
    mel = mel.squeeze(0)

# 🔧 Stretch time axis (CRUCIAL for audibility)
mel = mel.repeat_interleave(4, dim=0)

np.save("output_mel.npy", mel.cpu().numpy())
print("Saved mel:", mel.shape)
