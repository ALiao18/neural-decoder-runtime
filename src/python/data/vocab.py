import nltk
from nltk.corpus import cmudict

# ARPAbet phoneme set (39 phonemes)
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
    'B',  'CH', 'D',  'DH', 'EH', 'ER',
    'EY', 'F',  'G',  'HH', 'IH', 'IY',
    'JH', 'K',  'L',  'M',  'N',  'NG',
    'OW', 'OY', 'P',  'R',  'S',  'SH',
    'T',  'TH', 'UH', 'UW', 'V',  'W',
    'Y',  'Z',  'ZH'
]

# blank token at index 0, required by torch CTC loss default
BLANK_IDX = 0

# Phoneme to index (1-indexed, blank occupies 0)
PHONEME_TO_IDX = {p: i + 1 for i, p in enumerate(PHONEMES)}

# Index to phoneme
IDX_TO_PHONEME = {i + 1: p for i, p in enumerate(PHONEMES)}
IDX_TO_PHONEME[BLANK_IDX] = '<Blank>'

# total vocab size (39 phonemes + 1 blank)
VOCAB_SIZE = len(PHONEMES) + 1

# CMU pronuouncing dictionary
_cmudict = None

def _get_cmudict():
    global _cmudict
    if _cmudict is None:
        _cmudict = cmudict.dict()
    return _cmudict

def text_to_indices(text: str) -> list[int]:
    """
    Convert a sentence string to a list of phoneme indices. 
    Words not found in cmudict are skipped with a warning. 
    Stress markers are stripped (AA0, AA1, AA2 -> AA).
    """

    d = _get_cmudict()
    indices = []
    for word in text.lower().split():
        # strip punctuation
        word = ''.join(c for c in word if c.isalpha())
        if not word:
            continue
        if word not in d:
            print(f"Warning: Word '{word}' not found in CMU dictionary, skipping.")
            continue

        # take first pronunciation
        phonemes = d[word][0]
        for p in phonemes:
            # strip stress markers (digits at end)
            p_clean = p.rstrip('0123456789')
            if p_clean in PHONEME_TO_IDX:
                indices.append(PHONEME_TO_IDX[p_clean])
        
    return indices
            