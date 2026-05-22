_Project:_ Streaming Inference runtime for Intracortical Speech Brain Computer Interface decoding.  

_Motivation:_ BCI decoding requires fast, local inference speeds for data security as well as practicality. There are tons of research on different model architectures, but no benchmarked, model-agnostic inference library that a researcher could drop their trained model into. 

_Description:_ The engine will have numerically bit-identical outputs with the Python implementation of Willett Data, with 50, 95, 99 percentile latency benchmarks across sessions. Further info such as memory footprint and throughput at different beam lengths will also be recorded. 

_Target Benchmark:_ Brain-to-Text Benchmark 2024/5

_Stack:_ 
- C++ 17 as the backbone 
- pybind11 wrapper

_Features:_
1. Low latency (real-time feedback)
2. reliable (no Python GIL, no interpreter startup, no GC pauses)
3. Inspectable (structured logs, latency profiling, session tracking)
4. portable 
5. streaming (receive fixed size window spike data, maintain rolling buffer, segment, and inference, while emiting a partial transcript)

_Flow:_
1. _Layer 1 — Python (Training & Export):_
   - Load raw intracortical Utah array recordings (HDF5/NWB format)
   - Compute preprocessing constants (mean/std per channel per session)
   - Train RNN encoder with CTC loss, outputting phoneme logits
   - Export trained model to TorchScript (.pt file)
   - Output: .pt file + preprocessing constants (JSON)
   - Runs once. Everything downstream is C++.

2. _Layer 2 — C++ Runtime (Primary Deliverable):_
   - Loads .pt file via libtorch
   - Maintains a rolling buffer of incoming spike windows (20ms bins)
   - Preprocessing per window: spike binning, z-score normalization
   - Detects utterance boundaries (start/end of speech attempt)
   - Runs RNN forward pass on buffered windows
   - CTC beam search decode with n-gram language model (kenlm)
   - Emits partial transcript as windows accumulate
   - Logs per-request: session ID, latency per component
     (preprocessing / inference / decode / LM), p50/p95/p99 across sessions
   - Also logs: memory footprint, throughput at varying beam widths

3. _Layer 3 — Interface:_
   - _Streaming (primary):_ WebSocket endpoint
     - Client pushes fixed-size spike windows as they arrive
     - Server maintains session state and rolling buffer
     - Returns partial transcripts incrementally as speech is decoded
   - _Batch (benchmarking):_ POST /decode
     - Accepts pre-segmented spike array + session ID
     - Returns: transcript, latency breakdown, beam width used
   - _Response envelope:_

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






