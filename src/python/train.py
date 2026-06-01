import sys
sys.path.insert(0, './src/python')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time, perf_counter

from data.dataset import SpeechBCIDataset, BlockZScorer, collate_fn
from model.gru_ctc import SpeechBCIModel

# --- config ---
BASE_DIR       = '/Users/aliao/Documents/neural-decoder-runtime/'
DATA_DIR       = os.path.join(BASE_DIR, 'data/willet/competitionData/train')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'artifacts/checkpoints')
BATCH_SIZE     = 32
LEARNING_RATE  = 3e-4
NUM_EPOCHS     = 4
LOG_EVERY      = 10
BLANK_IDX      = 0
PROFILE_STEPS  = 3   # number of steps to print timing for, then disable

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

# --- data ---
print("Loading dataset...")
ds_raw = SpeechBCIDataset(DATA_DIR)
scaler = BlockZScorer(ds_raw.trials)
scaler.save('artifacts/constants.json')
ds = SpeechBCIDataset(DATA_DIR, scaler=scaler)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_fn, num_workers=0)

# --- model ---
device = (
    #torch.device('mps') if torch.backends.mps.is_available()
    torch.device('cpu')
)
torch.set_num_threads(12)
torch.set_num_interop_threads(4)
print(f"Device: {device}")

model = SpeechBCIModel(
    input_size=256,
    hidden_size=1024,
    num_layers=3,
    vocab_size=40,
    dropout=0.2,
).to(device)

ctc_loss   = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# --- training loop ---
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    start_time = time()
    model.train()
    epoch_loss  = 0.0
    num_batches = 0

    for step, (features, targets, input_lengths, target_lengths) in enumerate(loader):
        start_step_time = time()
        features = features.to(device)
        logits   = model(features)              # [T, B, 40] on MPS

        # --- CTC loss (CPU) ---
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        loss = ctc_loss(
            log_probs,
            targets,            
            input_lengths,      
            target_lengths,     
        )

        # --- backward ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- optimizer ---
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

        if step % LOG_EVERY == 0:
            end_step_time = time()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step:4d} | Loss {loss.item():.4f} | Step Time {end_step_time - start_step_time:.2f}s")

    avg_loss = epoch_loss / num_batches
    scheduler.step(avg_loss)

    end_time = time()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} complete | Avg loss {avg_loss:.4f} | "
          f"Duration: {end_time - start_time:.2f}s")

    if avg_loss < best_loss:
        best_loss = avg_loss
        path = os.path.join(CHECKPOINT_DIR, 'best.pt')
        torch.save(model.state_dict(), path)
        print(f"Saved best checkpoint (loss {best_loss:.4f})")

print("Training complete")