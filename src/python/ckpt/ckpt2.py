import torch
from model.gru_ctc import SpeechBCIModel

model = SpeechBCIModel()
x = torch.randn(25, 1, 256)   # [T=25, B=1, input_size=256]
out = model(x)

print(f"Input shape:   {x.shape}")
print(f"Output shape:  {out.shape}")
print(f"Expected:      torch.Size([25, 1, 40])")
print(f"Match: {out.shape == torch.Size([25, 1, 40])}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")