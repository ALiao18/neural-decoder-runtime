## Input
_What is the raw signal?_
The pipeline takes in Intracortical Utah Array recordings, fixed frequency sampled (20kHz in Willet et al.), preprocessed (threshold-crossed, binned 20ms windows), at 128 or 256 electrodes.Input data float32, shape [T, C]: T = number of 20ms bins, and C = channel count.

_What is one "trial"? (time window, label)_
1. _Streaming:_ Continuous [1, C] per 20ms rolling window. 
2. _Discrete:_ Pre-segmented [T, C] with known boundaries. 

_What preprocessing happens before the model sees data?_
  (binning window size, z-score: per channel? per session?)
  1. In Willet et al., a per channel spike threshold is defined 
  2. 20ms binning, so the final data looks like [T, C], where T = num. of 20ms bins, and C = electrode count. 
  2. z-score per channel per session. 

_What is the shape of the tensor the model receives? (time steps × channels, approximate values)_
[T, C]

## Output
- What does the model output per timestep? (phoneme logits, how many classes?)
- What does the CTC (Connectionist Temporal Classification) decoder turn that into? (a string)
- What does the runtime return to the caller? (transcript + metadata)

Decoder
├── preprocess(frame)         ← shared
├── encode(window)            ← shared  
├── decode_greedy(logits)     ← shared
├── decode_beam(logits)       ← shared
└── apply_lm(beam)            ← shared

BatchSession
└── run(full_array) → transcript + latency breakdown

StreamingSession
├── push(bin)                 ← accumulates into rolling buffer
├── maybe_emit_partial()      ← greedy decode on current window
├── detect_boundary()         ← activity threshold
└── flush() → final transcript + latency breakdown

## Session format
- What is a session?
Each session is one recording day. 

_Why does z-score normalization happen per session?_
Neurons fires at vastly different rates (more akin to the power law), if not normalized per channel, high firing neurons will dominate the normalization. z-scoring per session allows us to model drift, and treat each session as a brain state. 

## discrete contract
_What is the model?_
Discrete decoding uses a non-autoregressive model, such as the CTC (Connectionist Temporal Classification) model. 

_Decoding_
The production approach is prefix beam search on a growing sequence. Run encoder on a causal window and at each step extend the current beam of partial hypotheses. 

```
step t:
    encoder output: h_t (from causal window)
    extend all beam hypotheses by one CTC frame
    prune to top-K
    emit best current prefix as partial transcript
```

## Streaming contract
_What is the model?_
Streamining decoding uses an autoregressive model for next phoneme prediction. Industry standard (Whisper, Google's streaming ASR, on-device speech systems) uses a RNN-T (Recurrent Neural Network Transducer)

_What is the fixed window size the streaming interface accepts?_
The streaming interface accepts 20ms bins. 

_At 30kHz sampling, how many samples is that?_
samples = sampling rate * time, so 30k samples/s * 0.02 s = 600 samples

_What does the server need to maintain between windows? (rolling buffer, session state)_
1. rolling context buffer: continuously feeds encoder the last N bins of fixed size
2. utterance accumulator: starts from boundary start, resets on boundary end, feeds final beam search and CTC decoder
3. session state: z-score constants and Model hideen states. The session state is reset at the detected boundary. Future option could be having multiple hidden states for different locality (sentence level, session level, person level). 

_Boundary Detection_
simple: track mean acvitivy at bin t, if mean activity < x standard deviations for N consecutive bins, end of utterance
advanced: train a classifier for 3 different types of tokens: end of utterance, utterance, and begin of utterance

_Streaming Decoding_
1. Model must detect boundary. This is done via either energy/activity thresholding, i.e. activity below a certain threshold for n consecutive bins, or training a classifier later on. 
2. Model is trained and exposed to in runtime a 500ms context (25 bins). 
3. The encoder is ran every 5 bins (100ms)
4. The decoder continuously decodes partial transcripts greedily, and re-decodes the full utterance once boundary is detected. 

The streaming pipeline looks like
1. 20ms bin, appended to rolling buffer
2. if buffer has min_context bins, run encoder on sliding window of buffer
3. greedy decode and emit partial transcript
4. if boundary is detected, run full beam search and language model on the accumulated utterance
5. emit a final transcript
6. reset utterance accumulator