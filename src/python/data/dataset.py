import os
import numpy as np
import json
import scipy.io
from torch.utils.data import Dataset

class BlockZScorer:
    '''
    Computes and applies per-channel, per-block z-score normalization.
    Fits on all trials within each block 
    '''
    def __init__(self, trials: list):
        self.stats = {}   # block_idx -> {means: [...], stds: [...]}
        self._fit(trials)

    def _fit(self, trials: list):
        # group feature matrices by block
        blocks = {}
        for features, _, block in trials:
            blocks.setdefault(block, []).append(features)

        for block, matrices in blocks.items():
            stacked = np.concatenate(matrices, axis=0)   # [T_total, 256]
            means   = np.mean(stacked, axis=0)           # [256]
            stds    = np.std(stacked, axis=0)            # [256]
        
            # raise error for zero std
            if np.any(stds == 0):
                raise ValueError(f"Block {block} has zero std on channels {np.where(stds == 0)[0]}")
            
            self.stats[block] = {
                'means': means.astype(float).tolist(),
                'stds': stds.astype(float).tolist()
            }

    def transform(self, features: np.ndarray, block: int) -> np.ndarray:
        means  = np.array(self.stats[block]['means'], dtype=np.float32) 
        stds   = np.array(self.stats[block]['stds'], dtype=np.float32)
        return (features - means) / stds
    
    def save(self, path: str):
        # convert int to str for json serialization
        serializable = {str(k): v for k, v in self.stats.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent = 2)
        print(f"Saved z-score constants to {path}")

class SpeechBCIDataset(Dataset):
    '''
    Loads Willett competitionData .mat files

    Each trial is one sentence:
        features:   [T, n_channels] float32 (spikePow + tx1, area 6v only))
        label:      str
        block_idx:  int
    
    concatenated features are 256 channels total, with 
    1. the first 128 channels corresponding to spikePow from area 6v
    2. the next 128 channels correspond to tx1 from area 6v
    '''

    AREA_6V_COLS = slice(0, 128) # first 128 channels are area 6v

    def __init__(self, data_dir: str, scaler: BlockZScorer = None):
        self.trials = [] # list of dicts with keys features, label, block_idx
        self.load_data(data_dir)
        self.scaler = scaler

    def load_data(self, data_dir: str):
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.mat'):
                continue
            fpath = os.path.join(data_dir, fname)
            self._load_file(fpath)
    
    def _load_file(self, fpath: str):
        mat = scipy.io.loadmat(fpath)

        spike_pow = mat['spikePow']    # (1, num_trials) arr 
        tx1       = mat['tx1']         # (1, num_trials) arr 
        block_idx = mat['blockIdx']    # (num_trials, 1) uint8
        sentences = mat['sentenceText']# (num_trials, num_char) <U86

        num_sentences = spike_pow.shape[1] 

        for i in range(num_sentences):
            sp = spike_pow[0, i][:, self.AREA_6V_COLS].astype(np.float32) # (T, 128)
            tx = tx1[0, i][:, self.AREA_6V_COLS].astype(np.float32)       # (T, 128)
            features = np.concatenate([sp, tx], axis=1)
            
            label = sentences[i].tobytes().decode('utf-16').strip()
            block = int(block_idx[i, 0])

            self.trials.append((features, label, block))
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        features, label, block = self.trials[idx]
        if self.scaler is not None:
            features = self.scaler.transform(features, block)
        return self.trials[idx]







