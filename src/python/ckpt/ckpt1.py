from data.vocab import text_to_indices, IDX_TO_PHONEME

sentence = 'Nuclear rockets can destroy airfields with ease.'
indices = text_to_indices(sentence)
print(f"Indices: {indices}")
print(f"Phonemes: {[IDX_TO_PHONEME[i] for i in indices]}")
print(f"Length: {len(indices)}")