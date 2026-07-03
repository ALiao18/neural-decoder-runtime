# Week 1
1. z-score constants are computed on all 20 sentences per block, and not a held-out-safe partition. This incurs some label leakage before submission to compatitions, and should be revisited before first leaderboard submission
2. since we are building non-streamining mode first, we use bidirectional GRU for training and forward only for streamining inference. This means batch and streaming outputs will differ and will need to be fixed if streamining WER is significantly worse than batch WER. 

# Week 2
## statis library pattern for cpp runtime
**context:** needed to consider how to structure the cpp build so runtime logic can be unit tested independently of `main.cpp`
**options:** 1. single executable with all logic in main, 2. static library `ndr_runtime` with all runtime logic, linked by main executable and test binary
**decision:** `ndr_runtime`. All non-entrypoint logic lives in `ndr_runtime`, `main.cpp`, `ndr_tests` (to be implemented) both link agianst it
**why:** google test needs to call into runtime classes directly (`ModelRunner`, `ZScoreConstants`) without running the full binary. A flat executable makes this impossible, and this is the standard pattern for cpp projects. Risk is adding one extra CMake files to handle. 
**Revisit if:** likely never, this is early foundational structural decision

## Config validation fails early at construction
__context:__ `model_config.json` declares runtime invariants (`stride_bins <= context_bins`, `model_type=="ctc"`, `vocab=="arpabet"`) that the data contract requires the runtime to enforce
__options:__ either validate at first use or validate early in `ModelRunner` constructor before any inference call
__decision:__ validate early in constructor so that `ModelRunner` can throw immediately on construction if invariant is violated
__why:__ ease of early debugging for foreign users
__Revisit if:__ if runtime later supports hot-reloading configs mid-session, this will need a corresponding `validate()` call on reload

## Named-Scope latency timer, not hardcoded fields
__context:__ latency needs to be measured per component (preprocessing, inference, decoding, LM, total). Inference and preprocessing are implemented, decoding and LM aren't
__options:__ 1. hardcode named fields (`preprocessing_ms`, `inference_ms`, etc) on a struct, 2. generic named scope timer backed by a `label -> samples` map, where any component registers itself by calling `start(label)/stop(label)`
__decision:__ named scope timer `LatencyTimer` backed by `std::unordered_map<std::string, std::vector<double>>
__why:__ avoids refactoring the timer class everytime a new pipeline stage is implemented. Components not yet built are stubbed with empty `start/stop` scopes and report `null`, rather than showing fake 0.00. This runs the risk of string keyed lookups being marginally slower than struct field access, but time overhead is negligible
__revisit if:__ per-call overhead of the map lookup ever becomes measurable at higher iteration rates, if so, then consider switching to enum-keyed array

## Raw Pointer for Active Block Selection
__context:__ `ZScoreConstants` loads per block zscore stats (23 blocks, 256 channels) but 1 block active/session. `normalize()` is called once/20ms bin in streaming mode, where cost will matter
__options:__ 1. pass `block_id` into `normalize()` on every call, doing map lookup each time, 2. pre-select active block once at session init via `set_block()`, storing raw pointer into internal map for zero-lookup access on every subsequent call
__decision:__ raw pointer set once by set_block, dereferenced inside `normalize()`
__why:__ streaming mode will call `normalize()` every 20ms, eliminating a map lookup from hot path eliminates some latency. However, raw pointer into a map is only safe because `blocks_` is never mutated after construction (`load()` populates it once) if blocks were changed after construction, pointer would dangle
__revisit if:__ if runtime needs to support hot-swapping blocks mid-session without reconstructing `ZScoreConstants`

## Google Test via FetchContent
__Context:__ need a cpp testing framework, no prior GTest installation on the build machine
__options:__ 1. manually install GTest via homebrew or some manager, 2. use CMake's `FetchContent` to download and build GTest at configure time
__decision:__ `FetchContent`
__why:__ reproducibility, but requires network access at first CMake configure. This is assumed
__revisit if:__ if build times becomes a bottleneck

## parity verification via subprocess shell-out
__context:__ need to verify cpp pipeline produces numerically identical output to python pipeline on the same input
__options:__ 1. build pybind11 bindings now and call cpp runtime directly from python, 2. add a `--parity` CLI mode to the cpp binary that reads a fixture and prints logits as JSON; have pytest shell out to the compiled binary via `subprocess` 
__decision:__ subprocess shell-out for now, pybind11 deferred to later in the project 
__why:__ will spend dedicated time on pybind11, for now, we just want to verify pipeline works
__revisit if:__ once pybind11 exists, consider whether to migrate test to call bindings directly as needed, while keeping the subprocess as an end to end sanity check

## parity confirmed at 7.45e-09 max absolute difference
__context:__ E2E numerical comparison between the python reference pipeline and cpp runtime on identical input (`torch.manual_seed(0)`, block`"1"`, `[25, 256]` float32)
__note:__ `torch.jit.load` emitted a depreciatedwarning in favor of torch.export. We ignore this for now as torch.export is a different serialization path that would require reworking cpp loading.  