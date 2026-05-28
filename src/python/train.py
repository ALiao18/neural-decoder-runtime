import sys
sys.path.insert(0, './src/python')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SpeechBCIDataset, BlockZScorer, collate_fn
from model.gru_ctc import SpeechBCIModel

# --- config ---
BASE_DIR       = '/Users/aliao/Documents/neural-decoder-runtime/'
DATA_DIR       = os.path.join(BASE_DIR, 'data/willet/competitionData/train')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
BATCH_SIZE     = 8
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 10
LOG_EVERY      = 5
BLANK_IDX      = 0

os.makedirs(CHECKPOINT_DIR, exist_ok = True)

# --- data ---
print("Loading dataset...")
ds_raw = SpeechBCIDataset(DATA_DIR)
scaler = BlockZScorer(ds_raw.trials)
scaler.save('constants.json')
ds = SpeechBCIDataset(DATA_DIR, scaler=scaler)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0) # dataloader just doing collating, so num_workers=0

# --- model ---
device = (
    # torch.device('cuda') if torch.cuda.is_available()
    torch.device('mps') if torch.backends.mps.is_available() 
    else torch.device('cpu')
)
print(f"Device: {device}")

model = SpeechBCIModel(
    input_size=256,
    hidden_size=1024,
    num_layers=3,
    vocab_size=40,
    dropout=0.2
).to(device)

ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- training loop ---
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for step, (features, targets, input_lengths, target_lengths) in enumerate(loader): 
        features       = features.to(device)
        targets        = targets.to(device)
        input_lengths  = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        
        # forward pass on GPU
        logits    = model(features.to('mps')) # [T, B, 40]

        # CTC loss on CPU, MPS backend doesn't support CTC loss yet
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        loss = ctc_loss(
            log_probs.to('cpu'),
            targets.to('cpu'),
            input_lengths.to('cpu'),
            target_lengths.to('cpu'))
        loss.backward()

        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

        if step % LOG_EVERY == 0:
            print(f"Epoch {epoch} | Step {step:4d} | Loss {loss.item(): .4f}")

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch} | complete | Avg loss {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        path = os.path.join(CHECKPOINT_DIR, 'best.pt')
        torch.save(model.state_dict(), path)
        print(f"Saved best checkpoint (loss {best_loss:.4f})")

print("Training complete")