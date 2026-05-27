from data.dataset import SpeechBCIDataset
from data.zscorer import BlockZScorer

def main():
    path = '/Users/aliao/Documents/neural-decoder-runtime/data/willet/competitionData/train'
    ds = SpeechBCIDataset(path)
    features, label, block = ds[0]
    print(f"Total trials: {len(ds)}")
    print(f"Features shape: {features.shape}")
    print(f"Label: {label}")
    print(f"Block: {block}")
    print(f"dtype: {features.dtype}")

if __name__ == "__main__":
    main()