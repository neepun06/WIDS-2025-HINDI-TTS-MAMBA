# train.py

import torch
import os
from torch.utils.data import DataLoader
from preprocess.dataset import HindiTTSDataset
from preprocess.collate import collate_fn
from models.mamba_tts import MambaTTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 120
BATCH_SIZE = 2
LR = 3e-4

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset
ds = HindiTTSDataset(max_items=500)
loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# Model
model = MambaTTS(vocab_size=len(ds.tokenizer)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.L1Loss()

best_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for tokens, mel in loader:
        tokens = tokens.to(device)
        mel = mel.to(device)

        pred = model(tokens)          # (B, T, 80)
        pred = pred.transpose(1, 2)   # (B, 80, T)

        T = min(pred.size(2), mel.size(2))
        loss = loss_fn(pred[:, :, :T], mel[:, :, :T])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss,
            },
            os.path.join(SAVE_DIR, "best_model.pt"),
        )
        print(f"  Saved new best model at epoch {epoch}")

    # Always save last model
    torch.save(
        model.state_dict(),
        os.path.join(SAVE_DIR, "last_model.pt")
    )
