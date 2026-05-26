# Data Contract

**Version:** 0.1  
**Date:** May 2026  
**Status:** Pre-implementation

This document defines the interface between the researcher's pipeline and the neural decoder runtime. Everything upstream of this contract is the researcher's responsibility. Everything downstream is the runtime's responsibility.

## Guiding Philosophy

The runtime is preprocessing-agnostic. It does not make decisions about how neural signals should be transformed. It accepts whatever `float32` tensor the model was trained on and runs it. The only preprocessing the runtime performs internally is z-score normalization using constants computed externally by the researcher's pipeline and passed in at session initialization.

For example, the research can use threshold-crossing spike counts, or LFP band power, or spiking band power, or wavelet features, among other things. Up to discretion of user. 

## Signal Context 

The runtime is designed for invasive electrode arrays: Utah arrays (128–256 electrodes) and ECoG grids. Raw signals are typically sampled at 20–30 kHz, threshold-crossed or otherwise processed into features, and binned into fixed time windows before reaching this runtime. The runtime never sees raw voltage. All of this is upstream.

## Model Contract

The runtime accepts any TorchScript `.pt` file satisfying the following:

```
Input:  float32 tensor [T, C]
            T = number of time bins (context window)
            C = number of channels (electrodes)

Output: float32 tensor [T, V]
            T = number of time steps (same as input)
            V = vocabulary size (logits, pre-softmax)
```

The model must assume monotonic alignment between input frames and output tokens. CTC and RNN-T satisfy this. Global-attention encoder-decoders do not.

Model metadata is declared in a sidecar `model_config.json` at load time:

```json
{
  "model_type": "ctc",
  "input_channels": 256,
  "bin_size_ms": 20,
  "context_bins": 25,
  "stride_bins": 5,
  "vocab": "arpabet",
  "vocab_size": 40,
  "torchscript_path": "model.pt"
}
```

`model_type`: declares decoder strategy. `"ctc"` for MVP. `"rnnt"` as future extension.  
`input_channels`: must match C the model was trained on.  
`bin_size_ms`: informational. Enforced by the researcher's pipeline, not the runtime.  
`context_bins`: number of bins the encoder expects per forward pass (e.g. 25 = 500ms at 20ms bins).  
`stride_bins`: how often the encoder runs in streaming mode, in bins (e.g. 5 = every 100ms at 20ms bins). Must be ≤ `context_bins`. Runtime validates this at load time and raises if violated. Default 5.  
`vocab`: phoneme set. `"arpabet"` is the only supported value for MVP.  
`vocab_size`: 40 for ARPAbet (39 phonemes + blank CTC token).

## Session Contract

A session is one recording day. Neural signal statistics are non-stationary across sessions due to electrode drift, tissue response, and impedance changes. Z-score constants must be computed per session on the training partition of that session's data

Session state is initialized at the start of each session and persists until explicitly reset.

```json
{
  "session_id": "s1",
  "channel_means": [float, ...],
  "channel_stds":  [float, ...],
  "n_channels": 256
}
```

`channel_means` and `channel_stds`: arrays of length `n_channels`. Computed by the researcher's Python pipeline. Passed to the runtime via `constants.json` at session initialization.  
`n_channels`: must match `input_channels` in `model_config.json`. Runtime validates this at load time and raises if mismatched.

### Why Per-Channel

Each electrode sits in a different cortical location, contacts a different local neuron population, and has different impedance from manufacturing variance and tissue encapsulation. Baseline firing rates and signal amplitudes vary substantially across electrodes for hardware and anatomical reasons unrelated to the signal of interest. Per-channel z-scoring treats each electrode as an independent measurement device with its own baseline and scale. Per-session z-scoring accounts for global drift of the array over time. Both sources of variance are real and operate at different scales; both must be removed.

## Input Tensor

One time bin is one row: spike counts (or LFP power, or any other feature) across all C channels, aggregated over one bin window.

```
Shape:  [T, C]   float32
            T = number of 20ms bins
            C = number of channels

Example (500ms context, 256 channels):
    T = 25, C = 256  →  tensor shape [25, 256]
```

**Important:** T here is bins, not raw samples. At 30 kHz with 20ms bins, one bin aggregates 600 raw samples into one value per channel. That aggregation happens in the researcher's Python pipeline. The runtime receives `[25, 256]`, not `[600, 256]`.

The runtime applies z-score normalization internally:

```
x_normalized = (x - channel_means) / channel_stds
```

This is the only transformation the runtime applies. No filtering, smoothing, binning, or feature engineering.

## Streaming vs. Batch Input

### Batch Mode

Full pre-segmented trial delivered at once.

```
Input:  [T, C]  float32   (complete trial, T determined by trial length)
```

Used for EvalAI evaluation and correctness validation. Trial boundaries are known. The runtime runs full beam search + LM on the complete input and returns a final transcript.

### Streaming Mode

Continuous input, one bin at a time.

```
Input per step:  [1, C]  float32   (one 20ms bin)
```

The runtime maintains two internal buffers:

**Rolling context buffer** — fixed size `context_bins`, always contains the last N bins. Feeds the encoder continuously. Not reset at utterance boundaries.

**Utterance accumulator** — grows from detected boundary start. Reset at detected boundary end. Feeds full beam search when a boundary is detected.

## Boundary Detection

Utterance boundaries are detected via activity thresholding:

```
mean_activity_t = mean(spike_counts over C channels) at bin t
if mean_activity_t < threshold for N consecutive bins → end of utterance
```

`threshold` and `N` are configurable session parameters. Defaults are set empirically from the Willett dataset. 

Later, a trained classifier can be implemented. 

## Session State and Hidden State Policy

The hidden state is carried across bins within an utterance. At a detected boundary:

- Hidden state: hard reset to zero
- Utterance accumulator: flushed and reset
- Rolling context buffer: not reset (continuity of recent neural context is preserved)
- LM context: previous sentence token sequence is retained as prior for the next sentence's n-gram scoring

**LM context carry-over:** after a sentence is finalized, its decoded token sequence is prepended as context when scoring the next sentence's beam hypotheses. One sentence of context is retained. This is a standard conversational ASR technique and requires no additional model training.

**Future option:** exponential decay of hidden state during silence rather than hard reset. Deferred — hard reset is simpler and keeps batch/streaming outputs comparable for parity testing.

## Output Contract

Every response — streaming partial or batch final — returns the same envelope:

```json
{
  "session_id": "s1",
  "transcript": "hello world",
  "is_partial": true,
  "latency_ms": {
    "preprocessing": 4.2,
    "inference": 31.7,
    "decode": 9.1,
    "lm": 3.4,
    "total": 48.4
  }
}
```

`transcript`: current best hypothesis. Greedy for partials (streaming), beam search for finals (boundary detected or batch).  
`is_partial`: true for streaming partials, false for finals.  
`latency_ms`: per-component wall-clock time. Reported on every response. This is the primary engineering deliverable.

### Vocabulary and Output Dimension

```
Phoneme set: ARPAbet (39 phonemes)
CTC blank:   1 token
Total V:     40
```

One forward pass returns logits of shape `[T, 40]`. The CTC decoder converts this to a phoneme sequence. The LM converts the phoneme sequence to a word sequence.

---

## HTTP Interfaces

### Batch

```
POST /decode
Content-Type: application/json

{
  "session_id": "s1",
  "data": [[float, ...], ...]   // row-major [T, C]: data[i][j] = time bin i, channel j
}

Response: output envelope (is_partial: false)

`data` is a row-major nested array of shape `[T, C]`: outer index is time bin, inner index is channel. `data[i][j]` = bin i, channel j. The length of the outer array is T (number of bins). The length of each inner array is C and must match `input_channels` in `model_config.json`.
```

### Streaming

```
WebSocket /stream

Client → Server:  binary frame, [1, C] float32, row-major
Server → Client:  JSON output envelope per emission
```

---

## pybind11 Interface

The runtime is callable from Python without HTTP:

```python
import neural_decoder_runtime as ndr

session = ndr.Session("model_config.json", "constants.json")

# Batch — C-contiguous float32, shape [T, C]: axis 0 = time bins, axis 1 = channels
result = session.decode(np.ascontiguousarray(arr, dtype=np.float32))

# Streaming
session.push(bin_array)          # [1, C] float32
result = session.get_partial()   # returns latest partial or None
```

## Invariants the Runtime Enforces

1. `n_channels` in `constants.json` must match `input_channels` in `model_config.json`. Hard error if mismatched.
2. Input tensor dtype must be float32. Hard error otherwise.
3. Input tensor channel dimension must match `input_channels`. Hard error otherwise.
4. Z-score constants must be provided before any inference call. Hard error if missing.
5. `channel_stds` must contain no zero values. Hard error — division by zero in normalization.
6. `stride_bins` must be ≤ `context_bins`. Hard error if violated.

## Explicitly Out of Contract

- Raw voltage input. Binning and feature extraction happen upstream.
- Spike sorting or threshold computation. Researcher's pipeline.
- Any preprocessing beyond z-score normalization.
- Multi-language phoneme sets (MVP only).
- Sentence-level semantic context vectors (future direction).
- Boundary detection via trained classifier (future direction).
- RNN-T decoding (future direction — interface is designed to support it).