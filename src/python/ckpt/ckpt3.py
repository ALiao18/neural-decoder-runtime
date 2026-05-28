import sys
import os
sys.path.insert(0, './src/python')

from torch.utils.data import DataLoader
from data.dataset import SpeechBCIDataset, BlockZScorer, collate_fn

path_base  = '/Users/aliao/Documents/neural-decoder-runtime/'
path_train = os.path.join(path_base, 'data/willet/competitionData/train')
path_test  = os.path.join(path_base, 'data/willet/competitionData/test')

ds_raw = SpeechBCIDataset(path_train)
scaler = BlockZScorer(ds_raw.trials)
ds = SpeechBCIDataset(path_train, scaler = scaler)

loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
features, targets, input_lengths, target_lengths = next(iter(loader))

print(f"Feature shape:     {features.shape}")     # [T_max, 8, 256]
print(f"Targets shape:     {targets.shape}")      # [sum of phoneme lenghts]
print(f"Input lengths:     {input_lengths}")      # [8] varying T per trials
print(f"Target_lengths:    {target_lengths}")     # [8] varying phoneme lengths
print(f"Feature dtype:     {features.dtype}")    
print(f"Targets dtype:     {targets.dtype}")
print(f"Sum target lengths:{target_lengths.sum()}") # should match targets.shape[0]
print(f"Lengths match:     {target_lengths.sum() == targets.shape[0]}")
