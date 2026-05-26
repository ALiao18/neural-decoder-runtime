# Streaming Inference runtime for Intracortical Speech Brain Computer Interface decoding.  

_Motivation:_ 
BCI decoding requires fast, local inference speeds for data security as well as practicality. There are tons of research on different model architectures, but no benchmarked, model-agnostic inference library for streaming speech BCI decoding that a researcher could drop their trained model into. 

This project does not make decisions about how neural signals should be processed as that is the researcher's domain. It accepts whatever tensor a model was trained on and runs it fast. 

_Agnostics:_
1. __Device Agnostic:__ Tensors carry a device flag. The runtime runs on CPU, Apple Silicon (MPS), or CUDA without code changes. Latency benchmarks are measured on Apple M4 Max CPU, researchers with CUDA hardware can reproduce and compare independently
2. __Model Agnostic:__ The runtime accepts any TorchScript .pt file satisfying the input/output contract. It does not assume a specific architecture CTC is the MVP decoder, with RNN-T as an extension to be implemented later. 
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
WER matching Brain-to-Text 2024/5 (EvalAI leader board submission). This is the proof that the runtime produces correct output. 

The engine will be numerically equivalent within floating point tolerance with the Python implementation of Willett Data, with 50, 95, and 99 percentile latency benchmarks for each component (preprocessing, inference, decode, LM, total), measured across seessions. Further info such as memory footprint and throughput at different beam lengths will also be recorded. 

_Features:_
1. Low latency (real-time feedback)
2. reliable (no Python GIL, no interpreter startup, no GC pauses)
3. Inspectable (structured logs, latency profiling, session tracking)
4. portable 
5. streaming (receive fixed size window spike data, maintain rolling buffer, segment, and inference, while emiting a partial transcript)

## Future features
1. different modes for different hardware. A CPU, CUDA, or Metal flag can be implemented to speed up inference on different hardware. 
2. implement RNN-T later on for native streamning
3. Motor decoding, cursor control, and other BCI extensions
4. whisper-style global attention models
5. multi-language support





