import json
import torch
from model.gru_ctc import SpeechBCIModel
from data.vocab import BLANK_IDX, IDX_TO_PHONEME
import os

# --- config ---
CHECKPOINT_PATH   = 'artifacts/checkpoints/best.pt'
TORCHSCRIPT_PATH  = 'artifacts/model.pt'
CONFIG_PATH       = 'artifacts/model_config.json'

INPUT_SIZE   = 256
HIDDEN_SIZE  = 1024
NUM_LAYERS   = 3
VOCAB_SIZE   = 40
DROPOUT      = 0.2

VOCAB_PATH = 'artifacts/vocab.json'

# --- export vocab mapping, static ---
vocab = {
    'blank_idx': BLANK_IDX,
    'idx_to_phoneme': IDX_TO_PHONEME
}
with open(VOCAB_PATH, 'w') as f:
    json.dump(vocab, f, indent=2)
print(f"Saved vocab mapping to {VOCAB_PATH}")

# --- load model ---
print("Loading checkpoint...")
model = SpeechBCIModel(
    input_size  = INPUT_SIZE,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    vocab_size  = VOCAB_SIZE,
    dropout     = DROPOUT,
 )
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()

# --- export ---
print("Scripting model...")
scripted = torch.jit.script(model)
scripted.save(TORCHSCRIPT_PATH)
print(f"Saved TorchScript model to {TORCHSCRIPT_PATH}")

# --- write model_config.json ---
config = {
    'model_type': 'gru_ctc',
    'input_size': INPUT_SIZE,
    'bin_size_ms': 20,
    'context_bins': 25,
    'stride_bins': 5,
    'vocab': "arpabet",
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'vocab_size': VOCAB_SIZE,
    'dropout': DROPOUT,
    'torchscript_path': TORCHSCRIPT_PATH
}

with open(CONFIG_PATH, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved model config to {CONFIG_PATH}")