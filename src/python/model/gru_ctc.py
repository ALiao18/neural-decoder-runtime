import torch
import torch.nn as nn

class SpeechBCIModel(nn.Module):
    """
    GRU encoder + linear CTC projection

    Input:  [T, B, input_size]   (time-first, required by CTCLoss)
    Output: [T, B, vocab_size]   (time-first, required by CTCLoss)
    """

    def __init__(
            self, 
            input_size: int = 256,
            hidden_size: int = 1024,
            num_layers: int = 3,
            vocab_size: int = 40,
            dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = False, # time-first
            dropout = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # bidirectional doubles the output size, CTCLoss is in train.py
        self.projection = nn.Linear(hidden_size * 2, vocab_size) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      [T, B, input_size] float32

        Returns:
            logits: [T, B, vocab_size] float32
        """
        encoded, _ = self.encoder(x)         # [T, B, hidden_size * 2]
        logits = self.projection(encoded )   # [T, B, vocab_size]
        return logits