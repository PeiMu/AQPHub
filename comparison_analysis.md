# Gap Analysis: reproducility-transql_plus vs AQP_middleware vs Paper

## A. Missing from reproduction — mentioned in paper

### A1. Runner / Inference orchestrator (Section 5)
- Paper describes running full inference. AQP_middleware has `transql_runner.cpp` (C++) and `measure/run_prefill.py` (Python).
- **Status**: To be implemented as `transql_plus/runner.py` (Phase 4).

### A2. Autoregressive decoding with KV cache (Section 5)
- AQP_middleware `measure/run_decode.py`: after prefill, `l{l}_k_rope` and `l{l}_v` tables persist as KV cache. Each decode step does `INSERT INTO` to append new K/V rows, then attends over the full cache.
- **Status**: To be implemented in `scripts/run_decode.py`.

### A3. Weight pivot caching across runs (Section 4.3, implicit)
- AQP_middleware `run_prefill.py` pre-pivots all weight tables once (`weight_pivot_steps()`) and reuses them across repeated inference runs (`cached_wt_pivot=True`). Weight tables are static.
- The reproduction's `postopt.py` pivots weights fresh every time inside `pivoted_matmul_sql`.
- **Note**: This is a runtime optimization, not described explicitly in the paper. But required for fair benchmarks since weight pivot is a one-time cost.
- **Status**: To be implemented in `runner.py`.

### A4. Perplexity evaluation (Section 5)
- AQP_middleware `measure/run_perplexity.py`: WikiText-2 PPL computed entirely in SQL (logsumexp CTE to avoid fetching `seq_len x vocab_size` rows to Python).
- **Status**: To be implemented as `scripts/run_perplexity.py`.

### A5. MOE / Mixture-of-Experts support (Section 3.2)
- Paper claims 5 operator categories "cover" MOE architectures; mentions "builds indexes on expert identifiers."
- AQP_middleware has 3 extra op types: `TopKRouting`, `ExpertFFN`, `MoeAggregate`.
- The reproduction has `_is_moe_table()` schema stubs but no MOE operators.
- **Analysis**: TopKRouting requires `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` — a window function NOT expressible in the 5 paper categories. ExpertFFN and MoeAggregate DO decompose into the 5 categories, but the full MoE pipeline cannot run without TopKRouting.
- **Status**: Noted. Not implementing — see reproduction_note.md D10.

---

## B. Differences between reproduction and paper

### B1. Softmax: 2-step (reproduction/paper) vs 4-step (AQP_middleware)
- Paper Table 1: `Normalize_{exp, SUM, div}` — a 2-step pattern.
- AQP_middleware C++ and `run_prefill.py`: 4-step numerically stable (max, exp(x-max), sum, divide).
- Reproduction: default is 2-step per paper (Decision D5), with `stable=True` option for 4-step.
- **Risk**: 2-step overflows for real inference (exp of large scores). Production runs need `stable=True`.

### B2. Table fusion: flag column (reproduction/paper) vs row-offset range filter (AQP_middleware)
- Paper Section 4.2: "with a flag column distinguishing the projection type."
- AQP_middleware `FusedQKVSQL()`: splits via range filter on `out_col` (e.g., `WHERE out_col < 4096`).
- Reproduction: explicit flag column `('Q'/'K'/'V')` per paper's wording.
- **Note**: Integer range comparison is ~5-10% faster than string flag comparison, but the paper says "flag column."

### B3. Chunk index: raw offset (reproduction/paper) vs sequential (AQP_middleware)
- Paper Algorithm 1 line 10: `chunk_index: c` where `c` steps by `chunk_size` — raw offset.
- AQP_middleware: sequential chunk_id (0, 1, 2, ...).
- Reproduction: raw offset (0, 32, 64, ...) per Decision D7.
- **Note**: Both are correct — JOIN semantics are the same. The reproduction matches Algorithm 1 literally.

### B4. 1D norm schema: no row_index (reproduction) vs row_id=0 (AQP_middleware)
- Paper is silent on 1D norm storage.
- AQP_middleware: `(row_id=0, chunk_id, v)`.
- Reproduction: `(chunk_index, v)` — Decision D1, no redundant column.

### B5. PIVOT syntax
- Both AQP_middleware `run_prefill.py` and reproduction `postopt.py` use DuckDB native `PIVOT ... ON chunk_index IN (...)`.
- AQP_middleware unit tests also have a manual `MAX(CASE WHEN ...)` variant.
- **Aligned.**

---

## C. In AQP_middleware but NOT in paper

### C1. Benchmark harness / timing infrastructure
- `run_prefill.py`: per-step timing breakdown, peak RSS, DB size, JSON output.
- `run_llamacpp.sh`: llama.cpp baseline comparison. Paper states "the models used here are unquantized full-precision", so only F32 is relevant.
- `collect_results.py`: unified comparison table.
- `sample_prompts.py`: prompt generation from LMSys-Chat-1M.
- **Needed for Section 5 reproduction, not for Sections 3-4.**

### C2. Verification / diagnostic infrastructure
- 6 diagnostic scripts: `verify_weights_db.py`, `verify_single_layer.py`, `verify_ops_numpy.py`, `diag_attn.py`, `diag_layer0.py`, `diag_layers.py`, etc.
- **Not paper-required.**

### C3. ONNX topology.json import path
- AQP_middleware C++ has `BuildFromJSON()` for ONNX-sourced DAG topology.
- Reproduction has no JSON import — only hardcoded `build_llama3_8b()`.
- **Status**: To be implemented for generality.

---

## D. Paper claims verified

| Claim | Status |
|-------|--------|
| "38 to 7" materialized intermediates (Section 4.1) | Verified: test asserts exactly 7 per layer |
| Table fusion reduces tables (Section 4.2) | Verified: QKV + gate/up fusion tested |
| ROW2COL pivot trade-off (Section 4.3) | Verified: `tune_pivot.py` benchmarks pivot_width x subquery_width |
| Constant folding 1/sqrt(head_dim) into W_Q (Section 3.1.2) | Verified: both extraction paths apply it |
| 5 operator categories cover the forward pass (Section 3.2.1) | Verified for dense models (9 ops map to 5 categories) |

---

## E. Summary of action items

| Priority | Item | Paper section | Status |
|----------|------|---------------|--------|
| Must | Runner (prefill + decode) | Section 5 | Phase 4 |
| Must | Weight pivot caching | Section 4.3 (implicit) | Phase 4 |
| Must | Benchmark scripts | Section 5 | Phase 4 |
| Must | Perplexity script | Section 5 | Phase 4 |
| Should | JSON topology import | Section 3.1 | Phase 4 |
| Should | Softmax overflow demo | Section 3.2.1 | Phase 4 |
| Noted | DeepSeek MoE (TopKRouting gap) | Section 3.2 | Not implementing |
| Won't | Diagnostic scripts | -- | Not paper-required |
| Won't | JOB/DSB benchmarks | -- | AQP middleware scope |
