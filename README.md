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

## C++ Runtime (Implemented)

As of now, the following components are implemented and tested. This section will grow as decoding, the language model, and the server layer are built.

### `ModelRunner`
Loads a TorchScript `model.pt` via `torch::jit::load`, validates `model_config.json` invariants (`stride_bins ≤ context_bins`, `model_type == "ctc"`, `vocab == "arpabet"`) at construction time, and exposes a `forward()` method that accepts a `[T, C]` float32 tensor and returns `[T, V]` logits. Batch dimension management is handled internally.

### `ZScoreConstants`
Loads per-block z-score statistics from `constants.json` (23 blocks × 256 channels for the Willett dataset). `set_block(block_id)` pre-selects the active block once per session; `normalize()` then applies `(x - means) / stds` with zero per-call lookup cost. Enforces the session contract invariants: no zero-std channels, channel count must match `model_config.json`.

### `LatencyTimer`
A named-scope timer using `std::chrono::high_resolution_clock`. Any pipeline stage registers itself with `start(label)` / `stop(label)`; `report(n)` prints a JSON summary with p50/p95/p99 per component, matching the response envelope format below. Components not yet implemented (decoding, LM) report as unmeasured rather than a misleading `0.000`.

### Testing
- **Google Test** (`ctest`, run from `build/`): 10 passing cases covering `ZScoreConstants` — load correctness, invariant violations, and numerical parity against Python at `atol=1e-5`.
- **Parity test** (`pytest tests/python/test_parity.py`): runs the full z-score → forward pipeline in both Python and the compiled C++ binary on an identical fixed input, asserts `torch.allclose` at `atol=1e-5`. Currently passing at `7.45e-09` max absolute difference.

## Operating Modes
1. Batch Mode (primary for validation)
Pre-segmented trial input, with full beam search. Used for leaderboard submission and correctness validation. This is used to benchmark WER (Word Error Rate)
2. Streaming Mode (primary for deployment)
Continuous 20ms bin input with a rolling buffer and activity-threshold boundary detection. Partial signals are greddily decoded with beam search at boundary. 

Both modes shrae the same underlying decoder. Batch mode is streaming mode with known boundaries. 

## Correctness Guarantee
WER ≤ 9.7% on Brain-to-Text 2024 (EvalAI leaderboard). This is the proof that the runtime produces correct output, not just fast output. (not verified yet, this is the goal)
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

### Build
```bash
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu) # or nproc on Linux
```
CMake resolves the libtorch install path dynamically from your active Python environment (`python3 -c "import torch; print(torch.utils.cmake_prefix_path)"`), so no manual path configuration is needed — just make sure `torch` is installed in the active venv before running `cmake`.

### Run latency benchmark
From the repo root (paths in `model_config.json` are relative to repo root)
```bash
./build/neural_decoder_runtime artifacts/model_config.json
```
runs 1k iterations of z-score -> forward on a fixed input, prints a JSON latency report

### Run C++ tests
```bash
cd build
ctest --output-on-failure
```

### Run Python <-> C++ parity test
```bash
python3 tests/python/generate_parity_fixture.py # once, or after model changes
pytest tests/python/test_parity.py -v -s
```

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
├── artifacts/             ← model.pt, model_config.json, constants.json
├── src/
│   ├── python/
│   │   ├── data/          ← DataModule, NWB parsing, preprocessing
│   │   ├── model/         ← RNN encoder, CTC loss
│   │   ├── train.py
│   │   └── export.py      ← TorchScript export
│   └── cpp/
│       ├── runtime/       ← inference engine, rolling buffer
│       ├── third_party/   ← json.hpp (nlohmann)
│       ├── decode/        ← CTC greedy + beam search (not yet implemented)
│       ├── server/        ← WebSocket + REST endpoints (not yet implemented)
│       ├── bindings/      ← pybind11 Python bindings (not yet implemented)
│       └── main.cpp
├── tests/
│   ├── python/            ← pytest: data pipeline, model correctness
│   │   ├── test_parity.py
│   │   ├── generate_parity_fixture.py
│   │   ├── fixtures/
│   └── cpp/               ← Google Test: preprocessing, decode, server
│       ├── test_zscore.cpp
│       ├── CMakeLists.txt
│       ├── fixtures/
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




