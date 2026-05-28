from data.dataset import SpeechBCIDataset, BlockZScorer
import numpy as np
import os

def main():
    path_base  = '/Users/aliao/Documents/neural-decoder-runtime/'
    path_train = os.path.join(path_base, 'data/willet/competitionData/train')
    path_test  = os.path.join(path_base, 'data/willet/competitionData/test')

    ds_raw = SpeechBCIDataset(path_train)
    scaler = BlockZScorer(ds_raw.trials)
    scaler.save('constants.json')

    ds = SpeechBCIDataset(path_train, scaler=scaler)
    ds_test = SpeechBCIDataset(path_test, scaler=scaler)

if __name__ == "__main__":
    main()