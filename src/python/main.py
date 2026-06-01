import torch
from model.gru_ctc import SpeechBCIModel
import sys
sys.path.insert(0, '~/Documents/neural-decoder-runtime')

model = SpeechBCIModel()
model.load_state_dict(torch.load('checkpoints/best.pt', map_location='cpu'))
model.eval()

# load one trial
from data.dataset import SpeechBCIDataset, BlockZScorer
ds_raw = SpeechBCIDataset('competitionData/train')
scaler = BlockZScorer(ds_raw.trials)
ds = SpeechBCIDataset('competitionData/train', scaler=scaler)

features, label, block = ds[0]
x = torch.from_numpy(features).unsqueeze(1)  # [T, 1, 256]

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits.squeeze(1), dim=-1)  # [T, 40]

print(f"Mean blank probability: {probs[:, 0].mean():.4f}")
print(f"Mean non-blank probability: {probs[:, 1:].mean():.4f}")
print(f"Most predicted token: {probs.argmax(dim=-1).mode().values.item()}")