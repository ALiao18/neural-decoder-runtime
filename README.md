# Neural Decoder Runtime

A production-grade, open-source C++ streaming inference runtime for intracortical speech BCI decoding. 

Drop in a trained model. Get documented latency. 

## Motivation:
BCI decoding requires fast, local inference speeds for data security as well as practicality. There are tons of research on different model architectures, but no benchmarked, model-agnostic inference library for streaming speech BCI decoding that a researcher could drop their trained model into. 

This project does not make decisions about how neural signals should be processed as that is the researcher's domain. It accepts whatever tensor a model was trained on and runs it fast. 

## Design
```text
┌─────────────────────────────────┐
│        Python Pipeline          │
│  (researcher-owned)             │
│                                 │
│  Raw signal → features → bins   │
│  Compute z-score constants      │
│  Train model → export .pt       │
└──────────────┬──────────────────┘
               │  model.pt + constants.json
               ▼
┌─────────────────────────────────┐
│       C++ Runtime               │  ← primary deliverable
│                                 │
│  Load .pt via libtorch          │
│  Z-score normalization          │
│  Rolling buffer                 │
│  Boundary detection             │
│  RNN forward pass               │
│  CTC beam search + n-gram LM    │
│  Latency logging per component  │
└──────────────┬──────────────────┘
               │  WebSocket (streaming)
               │  HTTP REST (batch)
               ▼
┌─────────────────────────────────┐
│         Interface               │
│                                 │
│  Streaming: 20ms spike windows  │
│  → partial transcripts          │
│                                 │
│  Batch: POST /decode            │
│  → transcript + latency         │
└─────────────────────────────────┘
```

_Agnostics:_
1. __Device Agnostic:__ Tensors carry a device flag. The runtime runs on CPU, Apple Silicon (MPS), or CUDA without code changes. Latency benchmarks are measured on Apple M4 Max CPU, researchers with CUDA hardware can reproduce and compare independently
2. __Model Agnostic:__ The runtime accepts any TorchScript `.pt` file satisfying the input/output contract. It does not assume a specific architecture CTC is the MVP decoder, with RNN-T as an extension to be implemented later. 
3. __Preprocessing Agnostic:__ The runtime does not see the raw neural signal. It accepts a float32 tensor that the researcher's pipeline produced. Spike counts, LFP power, spiking band power, wavelet features are all acceptable. Minimal internal preprocessing: z-score normalization using externally computed constants, nothing else. 

_Constraints:_
1. __Signal Modality: invasive electrode arrays__ Utah arrays and ECoG are the primary targets, and Non-invasive modalities (EEG, MEG, fMRI) are out of scope. 
2. __Task: Streamining Speech Decoding__ Output is text. The runtime decodes neural activity to phonemes to words. Motor decoding, cursor control, and other BCI tasks are left for future implementation. 
3. __Decoder Family: Monotonic Alignment__ The runtime assumes the model eimts output sin temporal order with input frames. CTC and RNN-T satisfy this. Attention based encoder-decoders (whisper-style global attention) are left for future implementation
4. __Language: English__ Phoneme vocabulary is [ARPAbet](https://en.wikipedia.org/wiki/ARPABET) (39 phonemes + blank). Multi-language support is a future direction.

## Operating Modes
1. Batch Mode (primary for validation)
Pre-segmented trial input, with full beam search. Used for leaderboard submission and correctness validation. This is used to benchmark WER (Word Error Rate)
2. Streaming Mode (primary for deployment)
Continuous 20ms bin input with a rolling buffer and activity-threshold boundary detection. Partial signals are greddily decoded with beam search at boundary. 

Both modes shrae the same underlying decoder. Batch mode is streaming mode with known boundaries. 

## Correctness Guarantee
WER ≤ 9.7% on Brain-to-Text 2024 (EvalAI leaderboard). This is the proof that the runtime produces correct output, not just fast output.
C++ runtime output matches the Python reference pipeline within floating point tolerance given identical input.

## Latency Benchmarks (ms)
Measured on Apple M4 Max, CPU inference, Brain-to-Text 2024 dataset

| Component      | p50 | p95 | p99 |
| -------------- | --- | --- | --- |
| Preprocessing  |0.003|0.004|0.005|
| Inference      |5.103|5.179|5.219|
| Decoding (CTC) |     |     |.    |
| Language Model |     |.    |.    |
| Total          |.    |.    |.    |

Full benchmark report in `benchmarks/latency_report.md`

## Quickstart
Coming later. Will cover: build, model loading, serving your first session.

## Response Format
Each responses returns the following envelope
```{json}
{
  "session_id": "s1",
  "transcript": "hello world", 
  "is_partial": true, 
  "latency_ms": 
    {
      "preprocessing": 4.2,
      "inference": 31.7,
      "decode": 9.1,
      "lm": 3.4,
      "total": 48.4
    }
}
```

## Repository Structure
```
neural-decoder-runtime/
├── README.md
├── CMakeLists.txt
├── src/
│   ├── python/
│   │   ├── data/          ← DataModule, NWB parsing, preprocessing
│   │   ├── model/         ← RNN encoder, CTC loss
│   │   ├── train.py
│   │   └── export.py      ← TorchScript export
│   └── cpp/
│       ├── runtime/       ← inference engine, rolling buffer
│       ├── decode/        ← CTC greedy + beam search
│       ├── server/        ← WebSocket + REST endpoints
│       ├── bindings/      ← pybind11 Python bindings
│       └── main.cpp
├── tests/
│   ├── python/            ← pytest: data pipeline, model correctness
│   └── cpp/               ← Google Test: preprocessing, decode, server
├── benchmarks/
│   └── latency_report.md
└── docs/
    ├── architecture.md
    ├── data_contract.md
    └── design_log.md
```


## Future features
1. different modes for different hardware. A CPU, CUDA, or Metal flag can be implemented to speed up inference on different hardware. 
2. implement RNN-T later on for native streamning
3. Motor decoding, cursor control, and other BCI extensions
4. whisper-style global attention models
5. multi-language support
6. trained classifier for boundary detection 
7. consider LSL instead of WebSocket




