from dataset import SpeechBCIDataset, BlockZScorer
import numpy as np

def main():
    path = '/Users/aliao/Documents/neural-decoder-runtime/data/willet/competitionData/train'
    ds = SpeechBCIDataset(path)
    scaler = BlockZScorer(ds.trials)
    scaler.save('constants.json')

    # verify one trial
    features, label, block = ds[0]
    normalized = scaler.transform(features, block)
    print(f"Before - mean: {features.mean():.4f}, std: {features.std():.4f}")
    print(f"After  - mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
    print(f"Any NaN: {np.isnan(normalized).any()}")

if __name__ == "__main__":
    main()