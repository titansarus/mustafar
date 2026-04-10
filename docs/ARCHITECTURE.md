# Mustafar Repository Architecture & Internals

This document explains how the repository is structured, how the major modules interact, and what execution flow looks like for evaluation and kernel benchmarking.

## 1) High-level purpose

Mustafar implements sparse KV-cache pruning for LLM inference, with:

- Python model wrappers that modify Hugging Face Llama/Mistral attention behavior.
- A custom CUDA extension (`mustafar_package`) used by the kernel-accelerated path.
- Triton-based compression utilities for sparse KV representations.
- Scripts to run LongBench generation and compute benchmark metrics.

## 2) Repository layout

### Top-level orchestration and evaluation

- `pred_long_bench.py`: Main inference entrypoint for LongBench prediction generation. It parses args, loads a model variant, runs generation per dataset, and writes JSONL outputs under `pred/` or `pred_e/`.
- `eval_long_bench.py`: Reads prediction files and computes per-dataset scores using task-specific metrics from `metrics.py`. It supports LongBench and LongBench-E variants.
- `long_test.sh`: Convenience wrapper to run `pred_long_bench.py` with sparsity and model parameters.
- `mem_spd_test.py`: Latency and peak-memory script for generation-time kernel comparisons between Mustafar mode and dense HF mode.

### Model implementations

- `models/llama_mustafar_*.py`: Multiple Llama variants implementing different pruning combinations (e.g., key token-magnitude, value token-magnitude, output-aware variants).
- `models/mistral_mustafar_Kt_Mag_Vt_Mag.py`: Mistral variant.
- `models/llama_mustafar_kernel.py`: Kernel-integrated Llama path that calls the CUDA extension and compression helpers.
- `models/llama_think.py`, `models/llama_thinv.py`: ThinK-style structured pruning baselines.

### Kernel and native extension stack

- `kernel/csrc/*`: CUDA/C++ core for sparse SpMM and reduction kernels (`SpMM_API.cu`, kernel headers, tiling config).
- `kernel/build/Makefile`: Builds `SpMM_API.o` and shared library `libSpMM_API.so` targeting configurable SM architectures.
- `kernel/kernel_wrapper/*`: Torch extension wrapper that links compiled kernel objects into Python module `mustafar_package`.
  - `pybind.cpp` exposes `mustafar_key_formulation` and `mustafar_value_formulation` to Python.
  - `setup.py` builds the CUDA extension via `torch.utils.cpp_extension.CUDAExtension`.
- `kernel/compression.py`: Triton kernels and helper logic to build sparse bitmaps and pack non-zeros for batched key/value compression.

### Config and utility layer

- `config/model2path.json`, `config/model2maxlen.json`: model aliases and context-length mapping used by prediction script.
- `config/dataset2prompt.json`, `config/dataset2maxlen.json`: dataset-specific prompt templates and max generation tokens.
- `utils/process_args.py`: dataclass-based argument definitions (`ModelArguments`, `DataArguments`, `TrainingArguments`) and parser function `process_args()`.
- `metrics.py`: task metrics used by LongBench scoring (QA F1, ROUGE, retrieval/count/classification/code similarity).

## 3) Execution flow: LongBench pipeline

### 3.1 Prediction stage (`pred_long_bench.py`)

1. Parse arguments via `process_args()` dataclasses; key runtime knobs include `k_sparsity`, `v_sparsity`, `group_size`, `mode`, and benchmark split flag `e`.
2. Instantiate tokenizer/config based on model family (`LlamaConfig` vs `MistralConfig`).
3. Select operation mode:
   - `mustafar`: imports a selected Llama pruning class.
   - `mustafar-mistral`: imports Mistral Mustafar class.
4. Inject config controls like sparsity/group size into model config before `from_pretrained`.
5. Load dataset from `THUDM/LongBench` via `datasets.load_dataset`.
6. Prompt construction and truncation:
   - dataset-specific format template
   - middle truncation when prompt exceeds configured max length
   - optional chat template wrapping for chat-style models.
7. Generation and output: one JSON object per sample containing prediction, answers, class labels, and length; written to `pred/.../*.jsonl` or `pred_e/.../*.jsonl`.

### 3.2 Scoring stage (`eval_long_bench.py`)

1. Pick prediction directory based on `--e`.
2. For each dataset JSONL file, gather predictions/answers (and lengths for E split).
3. Route to the appropriate metric using `dataset2metric` mapping.
4. Write final results JSON to the run directory (`result.json`).

## 4) Model internals and attention modifications

Mustafar model files inherit Hugging Face modeling primitives and replace attention behavior.

### 4.1 Typical class structure

In `models/llama_mustafar_Kt_Mag_Vt_Mag.py`:

- `LlamaAttention_MUSTAFAR` stores standard attention metadata (`num_heads`, head dim, KV heads) plus Mustafar parameters (`k_sparsity`, `v_sparsity`, `group_size`, `residual_length`).
- Projection layers remain standard (`q_proj`, `k_proj`, `v_proj`, `o_proj`), preserving HF compatibility.

### 4.2 Magnitude pruning methods

- `dh_prune_key` and `dh_prune_value` compute element-wise magnitude thresholds per vector and zero out values below threshold.
- The code reshapes `[B,H,T,D] -> [-1,D]`, uses `torch.kthvalue` to derive threshold, creates a mask, then reshapes back.

This is the core pruning primitive for several variants.

### 4.3 Decode-time and group-aware logic

The same file includes older/experimental value pruning (`dh_prune_value_old`) with group-window and attention-score accumulation semantics, especially for GQA and decode windows. This indicates internals evolved through multiple pruning formulations and experiments.

## 5) Kernel-accelerated path internals

`models/llama_mustafar_kernel.py` is the bridge between model forward pass and custom sparse kernels.

- Imports `mustafar_package` (PyBind-exposed CUDA extension) and `kernel.compression` Triton helpers.
- Defines Mustafar attention classes similar to pure-Python variants, but intended for flash-attention-only workflow and native sparse operations.

### 5.1 Native extension boundary

The extension module exposes two functions:

- `mustafar_key_formulation`
- `mustafar_value_formulation`

These are callable from Python and implemented in CUDA wrapper code, enabling high-performance sparse attention sub-steps.

### 5.2 Build chain

1. Build core object in `kernel/build` (`SpMM_API.o`, optionally shared lib) via NVCC + cuBLAS/cuSPARSE link settings.
2. Install wrapper with `pip install -e kernel/kernel_wrapper`, where `setup.py` links `../build/SpMM_API.o` into `mustafar_package` extension.

### 5.3 Triton compression layer

`kernel/compression.py` contains Triton JIT kernels that:

- Calculate per-tile bitmaps of non-zero entries for key/value layouts.
- Compute compact counts with padding alignment.
- Pack non-zero values into compressed buffers using prefix sums and batch offsets.

These operations prepare sparse structures consumed by downstream kernels and reduce memory traffic for KV caches.

## 6) Configuration model and runtime knobs

Arguments defined in `utils/process_args.py` are the primary control surface:

- Sparsity: `k_sparsity`, `v_sparsity`
- Quantization placeholders: `k_bits`, `v_bits`, `k_quant_dim`, `v_quant_dim`
- Window/group behavior: `group_size`, `residual_length`
- Mode selection: `mustafar`, `dense`, `pruned`, `kivi(X)` style note in help text
- Benchmark split switch: `DataArguments.e`

This structure keeps scripts CLI-friendly while reusing Hugging Face parser conventions.

## 7) Evaluation internals

`metrics.py` mixes exact/fuzzy and language-specific metrics:

- QA F1 with normalization (`qa_f1_score`, `qa_f1_zh_score`)
- ROUGE-L for summarization (`rouge_score`, `rouge_zh_score`)
- Retrieval/count numeric heuristics
- Classification matching with fallback sequence similarity
- Code completion similarity via fuzzy ratio

These metrics are mapped per dataset in `eval_long_bench.py`, so each task is scored with a suitable objective function rather than one global metric.

## 8) Practical mental model of the repo

Think of the project as three layers:

1. **Research logic layer (Python models)**: defines *what* to prune and *when*.
2. **Compression/execution layer (Triton + CUDA)**: defines *how* sparse KV is encoded and computed efficiently.
3. **Experiment layer (scripts + configs + metrics)**: defines *how to run* reproducible comparisons and report scores.

Suggested reading order for onboarding:

1. `README.md` (workflow overview)
2. `pred_long_bench.py` (end-to-end inference path)
3. One model variant (e.g., `models/llama_mustafar_Kt_Mag_Vt_Mag.py`)
4. `models/llama_mustafar_kernel.py` + `kernel/compression.py`
5. `eval_long_bench.py` + `metrics.py` for result computation
