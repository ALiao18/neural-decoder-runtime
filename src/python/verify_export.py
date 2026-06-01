import torch
from model.gru_ctc import SpeechBCIModel

CHECKPOINT_PATH  = "artifacts/checkpoints/best.pt"
TORCHSCRIPT_PATH = "artifacts/model.pt"

INPUT_SIZE  = 256
HIDDEN_SIZE = 1024
NUM_LAYERS  = 3
VOCAB_SIZE  = 40
DROPOUT     = 0.2
TOLERANCE   = 1e-5

# -- load original model ---
model = SpeechBCIModel(
    input_size  = INPUT_SIZE,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    vocab_size  = VOCAB_SIZE,
    dropout     = DROPOUT,
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()

# --- load scripted model ---
scripted = torch.jit.load(TORCHSCRIPT_PATH)
scripted.eval()

# --- run identical input through both ---
torch.manual_seed(42)
x = torch.randn(100, 1, INPUT_SIZE) # [25, 1, 256]

with torch.no_grad():
    out_python   = model(x)
    out_scripted = scripted(x)

# --- verify ---
match = torch.allclose(out_python, out_scripted, atol=TOLERANCE)
max_diff = (out_python - out_scripted).abs().max().item()

print(f"Output shape (python): {out_python.shape}")
print(f"Output shape (scripted): {out_scripted.shape}")
print(f"Max absolute difference: {max_diff:.2e}")
print(f"torch.allcose(atol=1e-5): {match}")

if match:
    print(f"PASS - exported model matches Python model within tolerance")
else:
    print(f"FAIL - outputs diverge, fix before C++ runtime.")
