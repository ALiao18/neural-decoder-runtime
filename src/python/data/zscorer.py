import json
import numpy as np

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
            stacked = np.concatenate(matrices, axis=0)   # 