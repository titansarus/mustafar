# Mustafar — Architecture & Internals

> **Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference**  
> [Paper](https://www.arxiv.org/pdf/2505.22913)

This document explains the structure of the repository, the purpose of every file and directory, and how the different components interact at runtime.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Repository Layout](#2-repository-layout)
3. [Core Concepts](#3-core-concepts)
4. [Component Deep-Dive](#4-component-deep-dive)
   - 4.1 [models/ — Pruned Attention Variants](#41-models--pruned-attention-variants)
   - 4.2 [kernel/ — Sparse Attention Kernel](#42-kernel--sparse-attention-kernel)
   - 4.3 [utils/ — Shared Utilities](#43-utils--shared-utilities)
   - 4.4 [config/ — Runtime Configuration](#44-config--runtime-configuration)
   - 4.5 [longbench/ — Dataset Definition](#45-longbench--dataset-definition)
   - 4.6 [pred_long_bench.py — Inference Driver](#46-pred_long_benchpy--inference-driver)
   - 4.7 [eval_long_bench.py — Scorer](#47-eval_long_benchpy--scorer)
   - 4.8 [metrics.py — Metric Functions](#48-metricspy--metric-functions)
   - 4.9 [mem_spd_test.py — Latency / Memory Benchmark](#49-mem_spd_testpy--latency--memory-benchmark)
   - 4.10 [long_test.sh — Evaluation Entry-Point](#410-long_testsh--evaluation-entry-point)
5. [Data Flow: LongBench Evaluation](#5-data-flow-longbench-evaluation)
6. [Data Flow: Sparse Kernel Path](#6-data-flow-sparse-kernel-path)
7. [Pruning Strategy Matrix](#7-pruning-strategy-matrix)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Dependencies](#9-dependencies)

---

## 1. High-Level Overview

Mustafar is a research system that reduces the memory footprint and improves the inference speed of large language models (LLMs) by **pruning (sparsifying) the Key-Value (KV) cache** generated during autoregressive decoding.

The repository has two main evaluation tracks:

| Track | What it measures | Entry point |
|---|---|---|
| **LongBench Evaluation** | Accuracy impact of KV pruning across 16 long-context NLP tasks | `long_test.sh` → `pred_long_bench.py` → `eval_long_bench.py` |
| **Kernel Evaluation** | Wall-clock latency and peak GPU memory of the custom sparse attention kernel | `mem_spd_test.py` |

The pruning is applied **inside a custom attention module** that wraps the standard HuggingFace Llama / Mistral attention layer. All weight matrices are kept identical to the original pre-trained model; only the KV activations are zeroed out at runtime.

---

## 2. Repository Layout

```
mustafar/
├── ARCHITECTURE.md          # This file
├── README.md                # Quick-start guide
├── requirements.txt         # Python package pins
├── long_test.sh             # Shell wrapper for LongBench evaluation
├── pred_long_bench.py       # Main inference script (generates predictions)
├── eval_long_bench.py       # Scores predictions against ground truth
├── metrics.py               # Metric implementations (F1, ROUGE, …)
├── mem_spd_test.py          # Speed / memory benchmarking script
│
├── models/                  # Pruned model variants
│   ├── llama_mustafar_Kt_Mag_Vc_Mag.py   # Key token-wise Mag + Value channel-wise Mag
│   ├── llama_mustafar_Kt_Mag_Vc_Opa.py   # Key token-wise Mag + Value channel-wise OPA
│   ├── llama_mustafar_Kt_Mag_Vt_Mag.py   # Key token-wise Mag + Value token-wise Mag (default)
│   ├── llama_mustafar_Kt_Mag_Vt_Opa.py   # Key token-wise Mag + Value token-wise OPA
│   ├── llama_mustafar_Kt_Opa_Vt_Mag.py   # Key token-wise OPA + Value token-wise Mag
│   ├── llama_mustafar_kernel.py           # Kernel-accelerated variant (uses CUDA SpMV)
│   ├── llama_think.py                     # ThinK baseline: structured Key pruning
│   ├── llama_thinv.py                     # ThinK baseline: structured Value pruning
│   └── mistral_mustafar_Kt_Mag_Vt_Mag.py # Mistral-7B-Instruct-v0.2 variant
│
├── kernel/                  # Custom CUDA / Triton sparse attention kernel
│   ├── compression.py       # Triton kernels: bitmap encoding + compression
│   ├── csrc/                # Raw CUDA C++ source files
│   │   ├── SpMM_API.cu          # Public API: Key and Value sparse matmul launchers
│   │   ├── SpMM_Kernel.cuh      # Warp-level sparse matrix multiply kernel
│   │   ├── MatMulUtilities.cuh  # Shared memory tiling helpers
│   │   ├── MMA_PTX.cuh          # Inline PTX warp matrix-multiply-accumulate
│   │   ├── AsyncCopy_PTX.cuh    # Async shared-memory copy via PTX
│   │   ├── Reduction_Kernel.cuh # Split-K reduction kernel
│   │   └── TilingConfig.h       # Compile-time tile-size constants
│   ├── build/               # Build output directory
│   │   ├── Makefile             # nvcc build script → produces libSpMM_API.so
│   │   └── SpMM_API.cuh         # Shared header for the build
│   └── kernel_wrapper/      # PyTorch C++ extension that exposes the kernel to Python
│       ├── mustafar_wrapper.cu  # C++ wrapper that calls SpMM_API
│       ├── mustafar_wrapper.h   # Header declaring mustafar_key/value_formulation
│       ├── pybind.cpp           # pybind11 module registration
│       └── setup.py             # pip install -e . installs the `mustafar_package` module
│
├── utils/                   # Shared Python utilities
│   ├── process_args.py      # HuggingFace-style dataclass argument parsing
│   ├── data.py              # Dataset loading helpers (C4 / generic)
│   └── metrics.py           # (mirror of top-level metrics.py for utils imports)
│
├── config/                  # Static JSON look-up tables
│   ├── model2path.json      # model short-name → HuggingFace hub path
│   ├── model2maxlen.json    # model short-name → max token budget
│   ├── dataset2prompt.json  # LongBench task → prompt template
│   └── dataset2maxlen.json  # LongBench task → max generation tokens
│
├── longbench/               # LongBench dataset loading script
│   └── LongBench.py         # HuggingFace datasets-compatible loader
│
└── figs/
    └── top_level.png        # Architecture diagram used in README
```

---

## 3. Core Concepts

### KV Cache
During autoregressive generation, each transformer attention layer caches the Key and Value projections of all previously generated tokens. As sequence length grows, this cache becomes the dominant memory consumer. Mustafar **prunes low-magnitude entries to zero**, turning the KV cache into a sparse tensor.

### Unstructured Sparsity
Unlike structured pruning (which removes entire rows/columns), Mustafar zeros out **individual scalar elements** based on their absolute value (magnitude-based) or their contribution to the output (output-aware). The resulting sparse format is stored as a **bitmap + packed non-zero array** and fed into a batched SpMV kernel.

### Pruning Axes
Each cache can be pruned along two independent axes:

| Axis | Symbol | Meaning |
|---|---|---|
| Token-wise | `t` | The `num_to_prune` smallest elements **across the token dimension** per head and feature |
| Channel-wise | `c` | The `num_to_prune` smallest elements **across the hidden dimension** per token |

### Pruning Criteria
| Symbol | Method | Description |
|---|---|---|
| `Mag` | Magnitude-based | Use the absolute value of the activation as the importance score |
| `Opa` / `Opt` | Output-aware | Weight the activation magnitude by the corresponding attention score (Wanda-style) |

### Residual Window
To avoid pruning very recent tokens (which carry the most critical information for next-token prediction), the last `residual_length` tokens are **always kept in full precision** and concatenated back with the pruned historical KV cache before the attention computation.

---

## 4. Component Deep-Dive

### 4.1 `models/` — Pruned Attention Variants

Each file in `models/` implements a full `LlamaForCausalLM` (or `MistralForCausalLM`) class by subclassing the HuggingFace model and replacing the standard `LlamaAttention` module with a custom `LlamaAttention_MUSTAFAR`.

#### Class hierarchy (Llama variants)

```
nn.Module
└── LlamaAttention_MUSTAFAR     # custom attention with pruning logic
      used inside ↓
LlamaDecoderLayer_MUSTAFAR      # wraps attention + MLP
      used inside ↓
LlamaModel_MUSTAFAR             # stacks N decoder layers
      used inside ↓
LlamaForCausalLM_MUSTAFAR       # top-level model with lm_head
```

#### Key methods in `LlamaAttention_MUSTAFAR`

| Method | Purpose |
|---|---|
| `dh_prune_key(key_states, target_sparsity)` | Applies magnitude-based pruning to the Key cache along the hidden dim (token-wise) |
| `dh_prune_value(value_states, target_sparsity)` | Applies magnitude-based pruning to the Value cache along the hidden dim (token-wise) |
| `dh_prune_key_channel(...)` | Channel-wise variant (used in `Vc`/`Kc` files) |
| `dh_prune_value_old(...)` | Output-aware (Wanda-style) Value pruning — accumulated attention scores weight the activation magnitude |
| `calculate_sparsity(tensor)` | Diagnostic helper: returns fraction of zero elements |
| `forward(...)` | Overrides standard attention forward; calls prune methods and routes through FlashAttention |

#### Prefill vs. Decode phases

- **Prefill** (`q_len > 1`): The full prompt is processed in one shot. KV states for all prompt tokens are pruned immediately before being stored in the cache.
- **Decode** (token-by-token, `q_len == 1`): On each step, the new Key/Value is projected and added to the growing KV cache. The pruning is applied to the **historical** cache (optionally grouped, in the output-aware variant) and the last `residual_length` tokens are always appended unpruned.

#### Naming convention recap

```
llama_mustafar_{K_axis}{K_method}_{V_axis}{V_method}.py

K/V  = Key / Value cache
t/c  = token-wise / channel-wise pruning direction
Mag  = magnitude-based criterion
Opa  = output-aware criterion
```

#### Baseline variants

- **`llama_think.py`**: Implements the ThinK (Xu et al.) method — structured Key cache pruning driven by query-key dot products. Used as a comparison baseline.
- **`llama_thinv.py`**: ThinK applied to the Value cache.

#### Kernel-accelerated variant (`llama_mustafar_kernel.py`)

Adds the Triton-based bitmap compression step (`kernel/compression.py`) and calls the custom CUDA SpMV kernel via the `mustafar_package` Python extension. The compressed KV data is passed to `mustafar_key_formulation` / `mustafar_value_formulation` instead of using standard dense attention.

---

### 4.2 `kernel/` — Sparse Attention Kernel

The kernel directory has three layers:

#### Layer 1: Triton Compression (`compression.py`)

Two Python functions convert a dense pruned KV tensor into a compact sparse format:

```
convert_key_batched(inputs: Tensor[B, M, N])
convert_value_batched(inputs: Tensor[B, M, N])
```

Both functions perform the same two-stage pipeline using Triton JIT kernels:

**Stage 1 — Bitmap + count computation**

Each **64-element tile** of the input tensor is processed by one Triton program:
- `calculate_bitmap_key_batched` / `calculate_bitmap_value_batched`: iterates 64 elements, builds a 64-bit integer bitmap (bit `i = 1` if element `i ≠ 0`), and counts non-zeros (rounded up to the nearest multiple of 8, then halved for the packed float16 byte count).

The Key and Value versions differ in how they index tiles:
- **Key**: tiles are organized column-major (transposed layout) because Keys are accessed as `[B, N, M]` after a transpose.
- **Value**: tiles are organized row-major (standard layout).

**Stage 2 — Compaction**

- `compress_key_batched` / `compress_value_batched`: uses prefix sums on the per-tile bitmaps to compute write offsets, then scatters the non-zero values into a flat packed buffer.

**Outputs** returned per call:
| Output | Shape | Description |
|---|---|---|
| `bitmaps` | `[B, num_tiles]` | 64-bit integer bitmap per tile |
| `accum_counts` | `[B, num_tiles+1]` | Prefix-sum of non-zero counts (CSR-style offsets) |
| `packed_not_batched` | `list[Tensor]` | Per-batch flat array of packed float16 non-zeros |

#### Layer 2: CUDA SpMM (`csrc/`)

The CUDA source files implement a **batched Sparse Matrix-Vector / Matrix-Matrix multiplication** based on the Flash-LLM framework:

| File | Role |
|---|---|
| `SpMM_API.cu` | Top-level launch function: selects tile configuration and dispatches the appropriate kernel template |
| `SpMM_Kernel.cuh` | `Key_Kernel` / `Value_Kernel`: warp-level tiled SpMM with shared memory double-buffering |
| `MatMulUtilities.cuh` | Helper: stores partial results from registers back to shared memory tiles |
| `MMA_PTX.cuh` | Inline PTX `wmma`-style matrix multiply accumulate instructions |
| `AsyncCopy_PTX.cuh` | Async copy from global → shared memory (cp.async PTX) |
| `Reduction_Kernel.cuh` | Split-K reduction across partial sums |
| `TilingConfig.h` | Compile-time constants (`TILE_M`, `TILE_N`, `TILE_K`) |

The `build/Makefile` compiles `csrc/SpMM_API.cu` (which `#include`s all other `.cuh` headers) into `libSpMM_API.so` for SM architecture 89 (Ada Lovelace / RTX 4000-series) by default.

#### Layer 3: Python Extension (`kernel_wrapper/`)

| File | Role |
|---|---|
| `mustafar_wrapper.cu` | C++ functions `mustafar_key_formulation` and `mustafar_value_formulation` that unpack PyTorch tensors, call the CUDA API, and return an output tensor |
| `mustafar_wrapper.h` | Declarations of the two wrapper functions |
| `pybind.cpp` | pybind11 module (`mustafar_package`) exposing both functions to Python |
| `setup.py` | `pip install -e .` build script (uses `torch.utils.cpp_extension`) |

After installation, the model accesses the kernel as:

```python
import mustafar_package
output = mustafar_package.mustafar_key_formulation(bmp, NZ, idx, NZ_offset, B, M, K, Batch, groups)
```

---

### 4.3 `utils/` — Shared Utilities

| File | Contents |
|---|---|
| `process_args.py` | Three `@dataclass` argument groups parsed by `transformers.HfArgumentParser`: **`ModelArguments`** (model path, sparsity levels, group size, residual length, mode), **`DataArguments`** (dataset, batch size, LongBench-E flag), **`TrainingArguments`** (output directory, max length) |
| `data.py` | `TextDataset` (iterable PyTorch dataset that tokenises and chunks text), `get_c4()` and `get_loaders()` for the C4 calibration corpus |
| `metrics.py` | Mirror / re-export of the top-level `metrics.py` scoring functions |

---

### 4.4 `config/` — Runtime Configuration

Pure JSON files loaded at startup by `pred_long_bench.py`:

| File | Key → Value |
|---|---|
| `model2path.json` | `"Llama-2-7b-hf"` → `"meta-llama/Llama-2-7b-hf"` (HuggingFace Hub ID) |
| `model2maxlen.json` | `"Llama-2-7b-hf"` → `4096` (maximum context tokens for truncation) |
| `dataset2prompt.json` | `"narrativeqa"` → `"Read …\n\nQuestion: {question}\nAnswer:"` (prompt template with `{placeholders}`) |
| `dataset2maxlen.json` | `"narrativeqa"` → `128` (max new tokens to generate per sample) |

---

### 4.5 `longbench/` — Dataset Definition

`LongBench.py` is a HuggingFace `datasets`-compatible builder script that defines 20 long-context evaluation tasks. It is used as a local fallback; the main evaluation script loads datasets directly from the THUDM HuggingFace Hub (`THUDM/LongBench`).

The 20 tasks cover:
- **Multi-document QA**: hotpotqa, 2wikimqa, musique, dureader
- **Single-document QA**: narrativeqa, qasper, multifieldqa_en/zh
- **Summarisation**: gov_report, qmsum, multi_news, vcsum
- **Few-shot learning**: trec, triviaqa, samsum, lsht
- **Synthetic tasks**: passage_count, passage_retrieval_en/zh
- **Code completion**: lcc, repobench-p

---

### 4.6 `pred_long_bench.py` — Inference Driver

This is the central script that ties everything together for LongBench evaluation.

**Startup flow:**

```
parse args (process_args)
  │
  ├─ load model2path.json, model2maxlen.json
  ├─ detect model family (llama / mistral)
  ├─ load tokenizer (AutoTokenizer)
  │
  ├─ instantiate LlamaConfig / MistralConfig
  │     ↳ inject k_sparsity, v_sparsity, group_size, residual_length
  │
  └─ instantiate LlamaForCausalLM_MUSTAFAR / MistralForCausalLM_MUSTAFAR
        (selects which models/ file to import at line 139–150)
```

**Prediction loop (`get_pred`):**

```
for each sample in dataset:
  1. Format prompt using dataset2prompt template
  2. Tokenise; if > max_length truncate from the middle
  3. Build chat template (Llama-3 / Mistral instruct models)
  4. model.generate(max_new_tokens=dataset2maxlen[task])
  5. Decode output, post-process
  6. Write {"pred", "answers", "all_classes", "length"} to pred/<run_id>/<task>.jsonl
```

**Output path convention:**  
`pred/{model_name}_{max_length}_K_{k_sparsity}_V_{v_sparsity}/{dataset}.jsonl`

Example: `pred/Llama-2-7b-hf_4096_K_0.7_V_0.7/narrativeqa.jsonl`

---

### 4.7 `eval_long_bench.py` — Scorer

Reads the `.jsonl` files produced by `pred_long_bench.py`, applies the appropriate metric function per task, and writes `result.json` to the same directory.

```python
python eval_long_bench.py --model Llama-2-7b-hf_4096_K_0.7_V_0.7
```

Supports two modes:
- **Standard** (`--model`): scores all tasks in `pred/<model>/`
- **Extended** (`--e`): scores from `pred_e/<model>/` with length-stratified buckets (0–4k, 4–8k, 8k+)

---

### 4.8 `metrics.py` — Metric Functions

| Function | Task type | Method |
|---|---|---|
| `qa_f1_score` | QA (English) | Token-level F1 after normalisation |
| `qa_f1_zh_score` | QA (Chinese) | Token-level F1 with jieba tokeniser |
| `rouge_score` | Summarisation | ROUGE-L F1 |
| `rouge_zh_score` | Summarisation (Chinese) | ROUGE-L F1 after jieba segmentation |
| `classification_score` | Few-shot classification | Exact match / fuzzy fallback |
| `retrieval_score` | Passage retrieval (EN) | Exact paragraph number match |
| `retrieval_zh_score` | Passage retrieval (ZH) | Exact paragraph number match |
| `count_score` | Passage counting | Number extraction + match |
| `code_sim_score` | Code completion | fuzzywuzzy `ratio` similarity |

---

### 4.9 `mem_spd_test.py` — Latency / Memory Benchmark

A standalone script to measure the **wall-clock time** and **peak GPU memory** of the Mustafar sparse kernel versus the dense baseline. Key parameters are hardcoded at the top of the file:

```python
K_SPARSITY = 0.7
V_SPARSITY = 0.7
GROUP_SIZE = 32
BATCH_SIZE = 32
prompt_lenth = 300      # input tokens
output_length = 600     # new tokens to generate
num_repeats = 3
```

It runs one warmup pass, resets memory stats, then times `num_repeats` generation passes and reports:
- `used time`: average ms per generation call
- `peak mem`: peak GPU memory in GB

---

### 4.10 `long_test.sh` — Evaluation Entry-Point

A thin shell wrapper that forwards sparsity, model, and mode arguments to `pred_long_bench.py`:

```bash
bash long_test.sh <k_sparsity> <v_sparsity> <model_hf_path> <mode>

# Example:
bash long_test.sh 0.7 0.7 meta-llama/Llama-2-7b-hf mustafar
```

It always uses GPU 0 (`CUDA_VISIBLE_DEVICES=0`) and sets `group_size = residual_length = 32`.

---

## 5. Data Flow: LongBench Evaluation

```
long_test.sh
    │ arguments: k_sparsity, v_sparsity, model, mode
    ▼
pred_long_bench.py
    │
    ├─ [startup] Load config JSONs
    ├─ [startup] Build LlamaForCausalLM_MUSTAFAR with injected sparsity config
    │
    └─ for each dataset task:
           │
           ├─ load_dataset("THUDM/LongBench", task)
           ├─ for each sample:
           │      tokenise → truncate → build_chat → model.generate
           │          │
           │          └─ inside each attention layer forward():
           │                 prefill: prune full KV cache → store in past_key_value
           │                 decode:  append new KV → prune historical → concat residual
           │
           └─ write pred/<run_id>/<task>.jsonl
    ▼
eval_long_bench.py --model <run_id>
    │
    └─ for each .jsonl: compute metric → write result.json
```

---

## 6. Data Flow: Sparse Kernel Path

When `llama_mustafar_kernel.py` is used (mode = `'kernel'`), the attention computation replaces the standard dense FlashAttention call with:

```
query_states [B, H, 1, D]
key_states   [B, H, T, D]  ← pruned (sparse, contains zeros)
value_states [B, H, T, D]  ← pruned (sparse, contains zeros)

Step 1 — Triton compression (compression.py)
    convert_key_batched(key_states)
        → bitmaps_k, accum_counts_k, packed_nz_k

    convert_value_batched(value_states)
        → bitmaps_v, accum_counts_v, packed_nz_v

Step 2 — CUDA SpMV (mustafar_package)
    attn_weights = mustafar_key_formulation(
        bitmaps_k, packed_nz_k, accum_counts_k, NZ_offset_k,
        query_states, M, K, B, num_kv_groups
    )   → [B, H, 1, T]   (sparse Q·Kᵀ)

    attn_weights = softmax(attn_weights)

    output = mustafar_value_formulation(
        bitmaps_v, packed_nz_v, accum_counts_v, NZ_offset_v,
        attn_weights, Reduction_Workspace, M, K, B, num_kv_groups
    )   → [B, H, 1, D]   (sparse attn · V)

Step 3 — Output projection (standard nn.Linear)
    hidden_states = o_proj(output)
```

---

## 7. Pruning Strategy Matrix

The following table summarises all pruning configurations available and their corresponding model file:

| File | Key axis | Key criterion | Value axis | Value criterion |
|---|---|---|---|---|
| `llama_mustafar_Kt_Mag_Vt_Mag.py` | token | magnitude | token | magnitude |
| `llama_mustafar_Kt_Mag_Vc_Mag.py` | token | magnitude | channel | magnitude |
| `llama_mustafar_Kt_Mag_Vt_Opa.py` | token | magnitude | token | output-aware |
| `llama_mustafar_Kt_Mag_Vc_Opa.py` | token | magnitude | channel | output-aware |
| `llama_mustafar_Kt_Opa_Vt_Mag.py` | token | output-aware | token | magnitude |
| `llama_mustafar_kernel.py` | token | magnitude | token | magnitude + kernel |
| `mistral_mustafar_Kt_Mag_Vt_Mag.py` | token | magnitude | token | magnitude |
| `llama_think.py` | structured (ThinK) | query-driven | — | — |
| `llama_thinv.py` | — | — | structured (ThinK) | query-driven |

---

## 8. Key Design Decisions

### Extend HuggingFace, don't fork it
All model files import the original HuggingFace model classes and only override the attention module. Weight loading, generation, and tokenisation are all standard HuggingFace. This keeps the codebase small and makes it easy to adapt to new model versions.

### Residual window
Keeping the last `residual_length` tokens (default 128) uncompressed prevents catastrophic accuracy loss from pruning the immediately preceding context, which is most relevant to next-token prediction.

### Group-wise decode pruning (output-aware variants)
To compute the output-aware score during autoregressive decoding, attention weights from the last `group_size` tokens are accumulated in a `score_accumulator` buffer. Pruning is triggered only every `group_size` steps (lazy evaluation), amortising the overhead of sorting over the token dimension.

### Bitmap sparse format
A 64-bit integer bitmap encodes which of 64 adjacent elements are non-zero. This enables fast bitwise iteration in the CUDA kernel and is memory-efficient (1 bit per element overhead) compared to CSR or COO formats.

### Separate Key / Value kernels
Keys are stored and accessed transposed relative to Values, so the tile-indexing logic in the Triton and CUDA kernels differs between `Key` and `Value` variants, even though the mathematical operation is similar.

---

## 9. Dependencies

| Package | Version | Role |
|---|---|---|
| `torch` | 2.4.1 | Core tensor operations and autograd |
| `transformers` | 4.43.1 | HuggingFace model classes and tokenisers |
| `triton` | 3.0.0 | Triton JIT for the compression kernels |
| `flash-attn` | 2.7.3 | FlashAttention-2 for the prefill pass |
| `datasets` | 3.2.0 | LongBench dataset streaming |
| `accelerate` | 1.3.0 | `device_map="auto"` multi-GPU placement |
| `bitsandbytes` | 0.45.5 | Optional low-bit quantisation support |
| `numpy` | 2.2.2 | Bitmap shift precomputation |
| `jieba` | 0.42.1 | Chinese tokenisation for ZH metrics |
| `rouge` | 1.0.1 | ROUGE score computation |
| `fuzzywuzzy` | 0.18.0 | Fuzzy string similarity for code metric |
| CUDA 12.x + nvcc | — | Building `libSpMM_API.so` and the pybind extension |
