# Improve Node-Based Split — Middleware Overhead Optimization (Track B)

Split out of `new_split_strategy_analysis.md` on 2026-07-06. Scope: reduce the
middleware overhead of the **existing** node-based split loop without changing
split decisions or plan quality. This is NOT the new split strategy — the
boundary-policy / adaptive-depth work lives in `new_split_strategy_analysis.md`.

Status: **P2.0 DONE — P2.a REJECTED by pre-registered rule (§7). P2.c DONE,
all gates pass (§8). P2.b SKIPPED (user decision 2026-07-06 night: "P2.c
only, then Phase 2"). Track B closed** — remaining effort moves to Phase 2
(adaptive splitter, `new_split_strategy_analysis.md`). P2.c is **hard-coded**
(2026-07-06 late: flag + `AQP_P2C_DISABLE` removed; the three passes are
simply deleted from `RunBuiltInMiddleOptimizers` — no MiddleOptimize
consumer needs them, see `lingo-db-reuse.log` Track B for the audit).

---

## 1. Problem statement (measured)

- Middleware overhead on the BEST single-run config (uniform query_tpde +
  jit-cache=single-run-template) = **368 ms warm**, of which
  extract (`extract_next_sub-IR`) = **299 ms (81.5%)**. The often-quoted
  617/516 ms figures are the no-cache config.
- Component attribution (env-guarded `AQP_SPLIT_TIMING` timers, full JOB,
  repeat=2, cold+warm avg):
  - SplitIR ≈ 464 ms/pass, of which `MiddleOptimize` = 422 ms (**91%**);
  - inside MiddleOptimize, `join_order` ≈ 449 ms/pass (incl. bg-prep calls);
    the other 13 middle passes ≈ 70 ms/pass combined;
  - inside join_order, **`PlanEnumerator::SolveJoinOrder` (DP enumeration +
    cost model) ≈ 97%** — graph build/stats extraction and `Reconstruct`
    (6.5 ms) are noise;
  - `ConvertPlanToIR` = 29 ms/pass, double `QuerySplit::Split` = 8 ms,
    merges ≈ 4 ms.
- There is **NO SQL/IR round trip in node-based** — the plan persists as a
  DuckDB `LogicalOperator` across iterations. The cost is the **DP join-order
  enumeration re-run from scratch every split iteration**. Consequently, an
  IR-native port of the same DP does the same work and saves almost nothing;
  the win must come from **skipping or shrinking DP re-runs**.

## 2. Verified facts (carried from analysis-doc F6/F7)

- Exact DP for ≤ 12 relations (`THRESHOLD_TO_SWAP_TO_APPROXIMATE`,
  plan_enumerator.hpp:36), greedy above (plan_enumerator.cpp:472-480).
- Cost model = distinct-count ratio with `DEFAULT_SELECTIVITY = 0.2`
  (relation_statistics_helper.hpp:57).
- Temp tables (LogicalColumnDataGet) get **synthetic per-column
  distinct = cardinality, from_hll=false**
  (relation_statistics_helper.cpp:180-192) — the DP consumes only the temp's
  exact *cardinality*, which `UpdateRemainingIR` already injects. The −1.27 s
  exe advantage over none-split comes from exact cardinality alone; parity
  does NOT require temp distinct counts (Opt-3 verified empirically: injected
  min/max had no effect, commit 6f2ba61).
- Hard quality gate for ANY skip/incremental scheme: the current per-iteration
  re-plans deliver **−1.27 s exe vs none-split** (q_tpde); that advantage must
  hold (±50 ms).
- `MiddleOptimize` runs 14 passes on `LogicalOperator` with `Binder`,
  `TableFilterSet`, and storage-level HLL distinct stats
  (duckdb optimizer.cpp:469-598) — none of which SimplestIR has; porting to
  IR is not an option (and wouldn't shrink the DP anyway).

## 3. Implementation plan

| step | work | est. | depends on | verification gate | expected effect |
|---|---|---|---|---|---|
| P2.0 | **log-only skip-gate experiment**: per iteration, log (est card the DP used, actual temp card, whether re-running join_order changed the order). Decides everything below. | ~100 LOC, throwaway | — | full JOB, both jit levels; no behavior change | knowledge |
| P2.a | skip-if-unchanged gate: re-run `join_order` only when actual/est deviates > k (else reuse previous order) | ~100 LOC | P2.0 says hit rate high | (1) golden correctness; (2) exe unchanged (±50 ms) — the −1.27 s advantage must hold; (3) MW extract shrinks by skip-rate × 78–90% | up to −0.2–0.25 s on best config |
| P2.b | incremental DP (reuse DP subsets not containing the updated temp) — only if P2.0 shows orders DO change often | DuckDB-side, complex | P2.0 says hit rate low | same as P2.a | partial |
| P2.c | reduced middle pass list (13 non-join_order passes ≈ 70 ms/pass) | ~30 LOC | — | golden correctness; exe unchanged | −0.03–0.05 s |
| P1 | temp stats side-table + startup base-table snapshot — **optional** (no DP consumer; temps get synthetic distinct = cardinality). Only customer: JIT codegen — replace the on-demand `GetTempTableMinMax` rescan (duckdb_adapter.cpp:2213-2299). | ~700 LOC | — | stats spot-check; zero perf regression | small |

Combined bound: best-config MW is only 368 ms warm total; even perfect extract
elimination gives 5.30 → ~5.00 s. The stack is MW ~0.10–0.15 + jit ~0.33 +
exe ~4.59 ≈ **5.0–5.1 s** warm single-run unless exe itself improves (that is
the new splitter's job, not this track's).

Order: P2.0 first (one day, decides P2.a vs P2.b), then P2.a/c. P1 optional,
off the critical path.

## 4. Open uncertainties

1. **Skip-gate hit rate (P2.0)** — how often does re-running `join_order`
   after a temp-cardinality update actually change the join order? Decides
   whether P2.a captures most of the ~233–260 ms warm DP cost or whether
   incremental DP (P2.b) is needed. Unmeasured.
2. **Iteration-0 order parity** — any skip scheme must match the current
   re-plan orders from iteration 1 on (the −1.27 s exe gate above).

## 5. Instrumentation notes

- The `AQP_SPLIT_TIMING` env-guarded timers used for the attribution
  (SplitTimer in `src/split/node_based_splitter.cpp`, per-pass timing in
  duckdb `src/optimizer/optimizer.cpp` `RunOptimizer` + MIDDLE markers, phase
  timers in duckdb `src/optimizer/join_order/join_order_optimizer.cpp`) were
  **REVERTED on 2026-07-06** (temporary-code cleanup). P2.0 needs to re-add
  equivalent env-guarded logging; the numbers in §1 remain valid.
- HAZARD: libpg_query's pg_functions.hpp `#define fprintf(...)` silently
  deletes any `fprintf` in middleware TUs that include it — use iostream for
  diagnostics.
- `ReorderBeforeSplit` timing is already committed (flows into
  `extract_next_sub-IR` via `pending_extract_us_`,
  ir_query_splitter.cpp:948).

## 6. Measurement commands

```bash
cd measure
# uniform q_tpde + template cache (current best single-run)
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template off tpde
# tuned + template cache
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template off llvm job_result/tuned_per_subquery_node-based.json
# parse ONLY with the canonical parser:
# /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
```

Correctness: all 113 JOB queries vs `measure/duckdb_job_node-based_golden.txt`
(see measure/correctness_test.sh); CSV format unchanged (no new columns).

## 7. P2.0 results (2026-07-06 evening) — P2.a REJECTED

Setup: env-guarded (`AQP_P2_LOG`) log-only instrumentation, zero behavior
change (full-JOB diff vs golden = clean with env unset; all 113×2 pass with
it set). Full JOB, repeat=2, best config (q_tpde + tmpl-cache) AND
jit-level=none — **both configs produce byte-identical P2 logs** (the split
sequence is deterministic across jit levels in a suite run). Logs:
`/tmp/p2log_qtpde.txt`, `/tmp/p2log_nojit.txt`; analyzers `/tmp/p2_analyze.py`,
`/tmp/p2_canon.py` (throwaway).

Instrumentation (env-guarded; initially left in tree, **DELETED 2026-07-06
late** from both repos once P2.0/P2.c concluded — re-derive from
`lingo-db-reuse.log` Track B if Phase 2 needs est/act traces):
- duckdb `client_config.hpp`: `dbshaker_log_middle_join_order` (per-context,
  avoids the bg-prep race); `optimizer.cpp`: join-skeleton serializer +
  `P2JO same|changed pre= post=` around the JOIN_ORDER pass.
- middleware `node_based_splitter.cpp`: `P2Q`/`P2EST`/`P2ACT`/`P2TERM`;
  `ir_query_splitter.{h,cpp}`: bg-prep mirror saves the first extraction's
  estimate so the foreground can print it (bg thread must not print).
- GOTCHA found: BLOCK 2's `ReorderGet::ReorderTables` (duckdb
  reorder_get.cpp:283/451/506) rebuilds each group's joins with **fresh
  LogicalComparisonJoin nodes and drops estimated_cardinality** — the DP's
  estimate must be captured after the FIRST Split, before BLOCK 2.

Numbers (512 P2ACT injections; 417 followed by a join_order re-run, 95
terminal):

| metric | value | pre-registered bar |
|---|---|---|
| re-runs leaving join skeleton unchanged | **59.0%** | ≥ ~70% → fail |
| unchanged among deviation < 1.5× | **62.5%** (n=40) | ≥ ~90% → fail |
| unchanged among deviation < 4× | 69.4% (n=111) | — |
| mirror-only changes (commutative swap) | 1/418 — negligible | — |

Deviation does NOT predict order stability. Worse, stability is inversely
correlated with DP cost (same-rate by relation count of the re-run):
3 rel = 95.6%, 4 = 73.8%, 5 = 70.2%, 6 = 60.9%, 7 = 46.9%, 8 = 38.7%,
12+ = ~0–21%. The re-runs a gate could safely skip are the cheap small ones;
the expensive many-relation re-runs (which carry the ~233–260 ms) change
structurally >50% of the time — **they are earning their keep** (this is the
mechanism behind the −1.27 s exe advantage). Even a perfect skip-oracle
captures ≲ half the DP cost; a real k-gate captures far less and risks exe.

Consequence (pre-registered rule): **P2.a dead**. Remaining options:
- **P2.b** incremental DP: exactness-preserving reuse of DP-table entries for
  subsets not containing the new temp relation; complex DuckDB surgery,
  realistic ceiling likely ~100–150 ms of the 233–260 ms.
- **P2.c** reduced middle pass list: ~70 ms, ~30 LOC, low risk.
- **Stop Track B** and move to Phase 2 (adaptive splitter, −181 ms exe
  headroom), which competes for the same effort budget.

## 8. P2.c results (2026-07-06 night) — DONE, all gates pass

User decision after §7: **"P2.c only, then Phase 2"** (P2.b skipped — its
~100–150 ms ceiling doesn't justify complex DuckDB DP surgery vs Phase 2's
−181 ms exe headroom).

Design. Per-pass timing (P2CPASS atexit dump, gated on
`dbshaker_log_middle_join_order`; full JOB, repeat=2) put the non-join_order
middle passes at ≈55 ms/iter: column_lifetime 14.2 ms/iter (runs twice),
join_filter_pushdown 12.4, unused_columns 9.3, build_side 7.7,
reorder_filter 4.0 (join_order ≈234 ms/iter for scale). Extracted sub-plans
are re-optimized from SQL at execution time, so a middle pass matters only
if the split loop or the IR conversion consumes its output. Verified
non-consumers → **skip set = COLUMN_LIFETIME (both calls) + REORDER_FILTER +
JOIN_FILTER_PUSHDOWN** (projection maps are cleared by the split machinery,
subquery_preparer.cpp:846-864/:274-277, and never read by the IR converter).
KEPT: BUILD_SIDE_PROBE_SIDE (child swaps can change split decisions) and
UNUSED_COLUMNS (affects extracted SQL width).

Implementation (env-guarded, zero behavior change when unset):
- duckdb `client_config.hpp`: `dbshaker_reduced_middle_passes` (per-context —
  same bg-prep-race-free pattern as P2.0); `optimizer.cpp`
  RunBuiltInMiddleOptimizers: `if (!p2c_reduced)` around the three blocks.
- middleware `node_based_splitter.cpp`: `AQP_P2C_REDUCED` env →
  sets the flag on the foreground ctx in `Preprocess` and in
  `InitFromCrossQueryPrep` (covers the bg-prep-adopted context).

Gates (full JOB, best config q_tpde + tmpl-cache, 15 iters, canonical
parser warm avg; baseline/reduced CSVs `/tmp/p2c_{baseline,reduced}.csv`):

| component | baseline | reduced | delta | gate |
|---|---|---|---|---|
| MW | 0.388 | 0.358 | **−0.030** | shrink ∝ skipped passes ✓ (predicted ~30 ms) |
| extract_next_sub-IR | 0.320 | 0.289 | −0.031 | ✓ |
| jit | 0.343 | 0.345 | +0.002 | — |
| exe | 4.645 | 4.668 | +0.023 | within ±50 ms ✓ (max per-query +3.7 ms = noise) |
| total | 5.377 | 5.372 | −0.005 | — |

Correctness: 113/113 pass, ALL_CORRECT vs
`measure/duckdb_job_node-based_golden.txt` with `AQP_P2C_REDUCED=1`; also
clean with env unset (zero-behavior check). CSV format unchanged.

Verdict: works exactly as predicted (−30 ms MW), but the win is at the
run-to-run noise floor of the 5.4 s total. **Made default-on** (user
decision 2026-07-06 night): `P2cReducedEnabled` in
`node_based_splitter.cpp` now returns true unless `AQP_P2C_DISABLE` is set
(opt-out kept for A/B measurement). Re-verified with the new default:
113/113 pass + ALL_CORRECT vs golden.

**UPDATE 2026-07-06 late — HARD-CODED (flag removed).** Audit of all
MiddleOptimize consumers showed none needs the three passes: node-based is
the only live caller (sub-plans re-optimize from SQL); the duckdb in-engine
split loop is compiled out (`ENABLE_QUERY_SPLIT=0`) and re-runs all three
in `PostOptimize` before physical planning anyway; the middleware topdown
splitter never calls MiddleOptimize (uses full `Optimize()`). So the passes
were deleted outright from `RunBuiltInMiddleOptimizers` and
`dbshaker_reduced_middle_passes` + `P2cReducedEnabled`/`AQP_P2C_DISABLE`
removed from both repos. Re-verified: 113/113 + node-based golden diff
clean. Full audit trail: `lingo-db-reuse.log` (Track B section).

Track B totals: P2.a dead, P2.b skipped, P2.c banked (−30 ms, opt-in).
The remaining MW (≈0.36 s, of which ≈234 ms/iter is the DP itself) is the
price of the −1.27 s exe advantage (§7). **Track B closed; next: Phase 2.**
