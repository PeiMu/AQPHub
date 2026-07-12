# New Split Strategy — Consolidated Analysis & Design

Status: analysis phase complete; final-query tune-key bug fixed AND tuned configs
re-measured (2026-07-06) — F5 rewritten with post-fix numbers; §7 records the
re-measurement outcome. Second verification pass (2026-07-06 pm): all §6
uncertainties resolved or downgraded except iteration-0 parity (P2 gate);
§1 premise revised (node-based now BEATS none-split on exe); F2 fully verified
at code level (positional codegen → P4 is key-only); F6/uncertainty-4 measured
(topdown = plan quality, not round trip; `ReorderBeforeSplit` timing hole fixed
in ir_query_splitter.cpp:948); §5 plan re-ordered into two independent tracks.

**DOC SPLIT 2026-07-06 pm:** Track B (middleware-overhead optimization of the
EXISTING node-based loop — DP re-run avoidance, P2.x/P1) moved to
**`improve_node-based.md`**. This document keeps the analysis & design of the
NEW split strategy only. **ORACLE EXPERIMENT ADDED (F9/§8):** a better-than-
node-based splitter is real — per-query best over 6 coarse depth/threshold
configs = **−181 ms exe (−11.2%)** on the 14 heaviest queries.

Scope: this document consolidates all verification work done on the 12 points of
`design_for_a_better_splitter.md` against the limitations in
`split_strategy_limitation.md`, records the evidence, and turns the surviving
points into a feasible design + implementation plan. The new strategy operates on
SimplestIR (`third_party/IR_SQL_Converter/inc/simplest_ir.h`).

---

## 1. Problem statement & current state

- Split strategies ranked today: **node-based > relationship-center > topdown**.
- **[REVISED 2026-07-06, canonical parser on the Jul-2 CSVs]** The old premise
  "node-based loses +0.629 s to none-split, JFP asymmetry = 99.8% of the exec
  gap" is stale. Current numbers (warm, all 113):

  | config | MW | jit | exe | total |
  |---|---|---|---|---|
  | none-split q_tpde | 0.01 | 0.54 | 5.93 | 6.47 |
  | node-based q_tpde | 0.62 | 0.42 | **4.66** | **5.70** |
  | none-split q_llvm | 0.01 | 6.11 | 3.97 | 10.09 |
  | node-based q_llvm | 0.63 | 8.14 | **3.63** | 12.40 |

  Node-based now **beats** none-split on execution in both backends (−1.27 s
  with tpde): the cardinality-feedback plans more than repay the JFP loss.
  The AQP premise is delivering, not aspirational.
- Residual JFP-style loss (queries where node-based exe is *slower* than
  none-split): only **+217 ms over 34 queries** (q_tpde; worst single query
  7c +29 ms) / **+385 ms over 63 queries** (q_llvm; worst 7c +34 ms), versus
  −1.48 s / −0.72 s of wins on the rest. This is the hard ceiling for any
  boundary/JFP work (P3).
- Middleware overhead is now an *absolute* cost, not a gap-vs-none story:
  MW = 368 ms warm on the BEST config (q_tpde + tmpl-cache), extract =
  299 ms (81.5%), and ~78–90% of extract is the DP join-order enumeration
  re-run every iteration. Full attribution numbers and the elimination plan
  (P2.x/P1) now live in **`improve_node-based.md`** (Track B) — orthogonal
  to the new-strategy design in this document.
- The AQP premise: each executed subquery yields the **exact** cardinality of its
  temp table, which produces better join orders for the remainder than
  DuckDB's estimates — now confirmed by the exe numbers above.

---

## 2. Verified findings (each with evidence)

### F1. Subquery *structure* is fully predictable one step ahead; *physical plans* are not

Experiment: full JOB (113/113 correct) with `--spec-jit=recompile`,
`--jit-cache=single-run-strict`, query-jit TPDE. The speculation machinery is a
natural instrument: it predicts the next subquery's SQL before the current one
finishes, then compares against reality (`NormalizedSqlEquals`: sorted FROM items
+ sorted WHERE conjuncts, i.e. order-insensitive).

| counter | value | meaning |
|---|---|---|
| speculations | 194 | launched across 103 queries |
| **misses** | **0** | predicted next-subquery SQL never structurally wrong |
| **card_misses** | **151 (77.8%)** | assumed temp cardinality > 2× off from actual |
| hits | 24 | structure AND cardinality both right |
| not_ready | 19 | bg compile not finished in time |

Conclusions:
1. **Split boundaries can be committed early** (structure is deterministic given
   the current remaining IR) — enables pipelining of split analysis and
   speculative compilation.
2. **Physical decisions (join order, build/probe side, HT sizing) must stay
   late-bound** — DuckDB's cardinality estimates are >2× wrong in 78% of subquery
   launches. This is the strongest direct confirmation of the AQP premise on JOB.

Caveats: one-step-ahead only (k-step stability unmeasured); 194 specs < 533 total
iterations (not every iteration launches a spec); order-insensitivity means this
does NOT measure how often the join *order* changed — only that the *boundary*
didn't.

### F2. Cross-subquery JIT reuse is blocked by the cache key ONLY — the generated code is already position-independent (fully verified 2026-07-06)

Location correction: the JIT compiler lives in the **middleware**, not DuckDB:
`src/jit/ir_to_llvm.cpp`. Verified facts:

1. **Key**: `SerializeQjitPlanTemplate` (ir_to_llvm.cpp:5456-5503) keeps
   `source_table` name, `source_table_index`, `source_is_temp`, and per-column
   `table_index` in the template key.
   **CORRECTION (verified 2026-07-06, dump + code):** temp indices do NOT
   drift across repeats. `subquery_index` resets per query
   (`ResetQueryState`, duckdb_adapter.cpp:2677 — the splitter comment at
   ir_query_splitter.cpp:659 is stale) and binder table indices are
   per-statement (duckdb binder.cpp:224), so keys are **100% repeat-stable**
   (measured: 491/491 pairwise-identical keys across two full-JOB iterations,
   `AQP_DUMP_CACHE_KEYS`; includes all 28 threshold-abort queries). The
   cross-repeat recompile cause is `ClearObjCache()` at every iteration > 0
   for jit_cache < 3 (aqp_middleware.cpp:494-495, :773-774) — i.e. the
   *deliberate* "single-run" semantics, not the key. The key only blocks
   **within-pass cross-query** sharing.
2. **Code**: the generated qjit code addresses sources **positionally** —
   `sources[k]` where k = step index (:9029-9033), columns by position
   (`LoadColData(i)`, :9036-9038), hash tables by plan-local `ht_id`
   (:9050-9064). `source_table_index` appears *nowhere* in the emitted code.
   On a template hit, the params/source binding is rebuilt from the *current*
   plan (`BuildParamsBuffer(plan)`, :8835-8838). **Remapping is automatic.**
3. **No data-dependent baking**: row counts are runtime loads (comment :8792),
   min/max guards, membership gates, block-skip stats and HT pointers are all
   runtime-loaded; HT layout (`num_keys`, `tuple_size`, offsets) is in the key
   and cardinality-agnostic. Reusing code compiled for temp #3 on temp #7 is
   *safe and equally fast*.
4. **No config collisions**: `opt_tag` embeds `simd_isa_`, `fast_mode_`
   (compile backend), `cache_mode_`, skip_hash_cmp/batch_probe/prefetch flags
   (:5849-5851, :8800-8805) — per-subquery tuned configs keep distinct entries.
5. **Template-mode limitations P4 inherits (pre-existing, not new)**: expr-level
   SIMD filters disabled (:5876; reason: deterministic params layout,
   :5866-5868) and `SortFiltersByCost` disabled (:6749). Measured cost ≈ 0:
   tuned_strict exe 4.37 s vs tuned_tmpl 4.34 s. qjit ROF/SIMD (:9174) has no
   template guard (and is TPDE-excluded anyway).

Shape statistics — **REMEASURED 2026-07-06 from the actual key dump**
(`AQP_DUMP_CACHE_KEYS`, full JOB in-suite, debug build, uniform q_tpde + tmpl,
repeat=2; supersedes the `--print-sql` counts, which used a different
standalone split — see F8):
- Per warm pass: **491 foreground (C2) compile requests** (+ ~112 background
  prep compiles, tag C1, off the critical path); 81 hits today (**16.5%**).
- Canonical simulation (first-occurrence index renumbering + temp-name
  normalization): 410 → 336 actual compiles; hit rate → **31.6%**,
  **74 newly-shared compiles per pass = 12.7% of total plan bytes**.
- **The newly-shared compiles are all small**: plan length med 266, max 607
  (vs all-plans med 288, max 2850). Every expensive plan — the big finals of
  9d/19d/24a/8c/23b/29c/16b, len 842–2850, the ones costing 32–94 ms each in
  tuned mode — is **canonically unique**: P4 saves nothing on them.
- Temp-referencing (60.3% by the old count; 82.3% of dump entries) is
  irrelevant to cross-repeat cost (keys are repeat-stable, see point 1).

Consequence: P4 is a **pure key-canonicalization** change (canonical temp/table
renumbering inside `SerializeQjitPlanTemplate` + the loc/expr serializers,
likely <200 LOC, middleware-only), but its warm win is only the within-pass
sharing: ≈ 13–15% of the warm jit column ≈ **40–50 ms/pass** on uniform
q_tpde_tmpl (0.33 s jit), less on tuned (per-subquery opt_tags fragment the
canonical classes). Verification instrument: `AQP_DUMP_CACHE_KEYS` (debug,
:8813-8832). Recovering the *cross-repeat* compile cost would require not
clearing the cache between iterations — which is by definition the already
measured full-cache class (3.65/4.48), not a P4 outcome.

### F3. Parallel subquery execution: no thread explosion, but contention is real and gains are small

- DuckDB's `TaskScheduler` is **per-DatabaseInstance**: all connections share one
  worker pool (12 threads here). Two concurrent `Query()` calls interleave tasks;
  DuckDB never spawns per-query threads.
- BUT the middleware query-jit executor has its own morsel scheduler + OpenMP
  kernels; `g_bg_active_threads` throttles only the middleware kernels, not
  DuckDB workers → mixing a query-jit subquery with a DuckDB subquery in parallel
  oversubscribes.
- Measured contention from existing data: a **single** bg compile thread
  (spec-jit) costs +16% exec on query_full (3.63→4.20 s) and +2.8% on q_tpde
  (4.62→4.75 s). Spec-jit still wins on query_full total (12.40→8.91 s, compile
  hidden) but *loses* on tpde total (5.32→5.68 s, compile already cheap).
- Independent (bushy) subquery pairs are a minority of JOB; heavy subqueries
  saturate all cores alone.

Verdict: **drop generic parallel subquery execution** from the design. Keep only:
(a) background speculative *compilation* (already exists), (b) optionally
overlap a *tiny* side subquery (< a few ms) with a big one — low priority.

### F4. Hybrid flag-tiering on fixed boundaries does not pay (experiment, corrected)

Setup: 6 heavy queries (10c, 16d, 19d, 30b, 6d, 8c), node-based, storage plan,
spec-jit recompile. Hybrid = interp on subquery 0, operator_tpde middle,
query_tpde final (via `--tune-config`); measured after fixing the final-key bug
(§6). Warm = mean of iters 1–4; cold = iter 0. All ms.

| query | uniform query_tpde (warm/cold) | hybrid (warm/cold) | interp (warm/cold) |
|---|---|---|---|
| 10c | 122 / 562 | 122 / 530 | 250 / 303 |
| 16d | 70 / 95 | 73 / 85 | 100 / 110 |
| 19d | 215 / 324 | 216 / 305 | 364 / 373 |
| 30b | 38 / 274 | 55 / 297 | 58 / 61 |
| 6d | 87 / 93 | 112 / 112 | 151 / 153 |
| 8c | 169 / 198 | 170 / 195 | 400 / 420 |
| **TOTAL** | **701 / 1546** | **748 / 1524** | **1323 / 1420** |

- Hybrid loses 47 ms warm, saves only 22 ms cold. 6d isolates the cause: interp
  on the first subquery alone costs +25 ms warm — TPDE compile is so cheap that
  skipping it saves less than the interpretation penalty.
- Correction of a prior belief: **interp-first is NOT the current default
  behavior**; uniform flags apply to all iterations, and the tuner data shows
  query_tpde wins 75.9% of subqueries *including first subqueries*.
- The real "hybrid split" idea is therefore about **boundaries**, not flags:
  fewer/better boundaries (less JFP loss, big-table-last, compile hidden behind
  earlier execution). Flag-tuning on fixed boundaries cannot recover what a bad
  boundary already lost.

### F5. Per-subquery tuning DOES compose for execution; its deficit is pure compile time (re-measured post-fix)

History: the pre-fix measurement (tuned 6.90 s vs uniform query_tpde 5.70 s)
suggested "greedy per-subquery composition fails". That number was contaminated
by the final-query tune-key bug (§6): on all **28 threshold-abort queries**

```
10c 12a 12b 12c 13b 13c 13d 15c 15d 16a 16b 16c 16d 17a 17e
19c 19d 23a 23b 23c 24a 29c 2d 33a 8c 8d 9c 9d
```

(from a full-JOB debug pass counting `[NodeBased] Cardinality threshold
exceeded`; `kSplitCardThreshold` = 1M, standalone only) the final — heaviest —
query ran with stale flags.

**Post-fix re-measurement** (all 8 tuned configs re-run 2026-07-06). Fix
validation: the abort group improved by exactly the predicted amount, the rest
did not move (warm sums, ms):

| group | pre-fix tuned | post-fix tuned | Δ | uniform q_tpde |
|---|---|---|---|---|
| abort (28) | 3340 | 2305 | **−1035** | 2167 |
| rest (85) | 3564 | 3551 | −13 (noise) | 3533 |

Biggest per-query movers: 8c 385→164, 19d 329→200, 9d 212→108, 10c 218→121 —
each landing on top of its uniform-q_tpde value.

Config-level picture after the fix (MW / jit / exe / total in s; parsed with the
canonical `plot_middleware_jit.py` parser — warm mean of iters 5–14,
components: MW = all non-exe/non-jit columns, exe = execute_sub-SQL +
final_exe, jit = jit_compile + jit_compile_final):

| config | MW | jit | exe | total | abort28 (ms) | rest85 (ms) |
|---|---|---|---|---|---|---|
| **pre-fix tuned** | 0.60 | 0.51 | 5.80 | 6.90 | 3340 | 3564 |
| tuned | 0.61 | 0.81 | **4.43** | 5.86 | 2305 | 3551 |
| uniform query_tpde | 0.62 | 0.42 | 4.66 | **5.70** | 2167 | 3533 |
| tuned_tmpl | 0.37 | 0.83 | **4.34** | 5.55 | 2177 | 3368 |
| uniform q_tpde_tmpl | 0.38 | 0.33 | 4.59 | **5.30** | 1937 | 3363 |
| tuned_tmpl_srec | 0.38 | 0.78 | 4.43 | **5.59** | 2129 | 3460 |
| uniform q_tpde_tmpl_srec | 0.39 | 0.54 | 4.74 | 5.67 | 2008 | 3666 |

jit-cache=full is a **separate comparison class** (persistent disk cache across
runs — steady-state numbers are not comparable to the single-run regimes above,
where "current best" = q_tpde_tmpl 5.30 s). Within the full-cache class:

| config (full-cache class) | MW | jit | exe | total | abort28 (ms) | rest85 (ms) |
|---|---|---|---|---|---|---|
| tuned + full | 0.03 | 0.01 | 4.44 | **4.48** | 1775 | 2709 |
| uniform q_tpde + full | 0.03 | 0.00 | 4.70 | 4.73 | 1801 | 2929 |
| uniform query_full + full | 0.03 | 0.00 | **3.62** | **3.65** | 1410 | 2240 |

(Cold iter-0 sums in the full-cache class: query_full 11.1 s, tuned 6.2 s,
q_tpde 5.9 s — full-LLVM's dominance exists only once the disk cache is warm.
The full-cache rows are used here as a *measured bound* for what P4/P5 could
approach within the single-run regimes, not as a competitor to 5.30.)

Conclusions (replacing the pre-fix one):
1. The fix behaves exactly as predicted: pre→post improvement is entirely in
   the 28 abort queries (−1035 ms); the other 85 moved −13 ms (noise). The exe
   component dropped 5.80 → 4.43 s (finals no longer run with stale flags)
   while jit rose 0.51 → 0.81 s (finals now actually compile as tuned).
2. **Per-subquery tuning composes for execution**: tuned has the best exe in
   every regime (4.34–4.44 vs 4.59–4.70 uniform tpde, ≈ −0.25 s), refuting the
   pre-fix "greedy composition fails" conclusion.
3. Its total-time deficit vs uniform tpde is **entirely compile time**
   (jit 0.8 vs 0.33 s): the tuner picks query_full/LLVM where O2 code wins on
   exec, and that compile is neither hidden nor cached in the plain regime.
4. **When compile is hidden (spec-jit), tuned wins** (5.59 vs 5.67). In the
   separate full-cache class, tuned beats uniform tpde (4.48 vs 4.73) but
   *uniform query_full* dominates outright (3.65 s) — with free compile you
   should JIT everything at O2.
5. **Tuning is regime-dependent**: the tuned JSON was selected under
   compile-inclusive, no-cache tradeoffs; under full cache the optimal "tuned"
   degenerates to query_full everywhere. Point 8 (re-keyed/regime-aware
   tuning) is upgraded from "defer" to "worth doing alongside P4".
6. Why doesn't the template cache absorb tuned's compile (0.83 s with tmpl)?
   **CORRECTED 2026-07-06 (was wrong):** not the temp indices — keys are
   repeat-stable (F2.1). The cache is deliberately cleared between iterations
   (`ClearObjCache()`, single-run semantics), so every miss recurs every
   iteration. **P4 cannot recover this**; keeping the cache across iterations
   *is* the full-cache class (already measured: tuned_full 4.48, query_full
   3.65). Warm jit decomposition from the existing CSVs: uniform q_tpde_tmpl
   335 ms = 208 sub + 127 final; tuned_tmpl **1031 ms** = 727 sub + 304 final,
   concentrated in a few canonically-unique big compiles (9d 94 ms,
   19d 68 ms, 24a 62 ms, 8c 49 ms, 13d 42 ms, 16b 32 ms — per iteration).
   P4's within-pass sharing (F2) touches none of these; its realistic warm
   gain is ~40–50 ms (uniform) / ~30–60 ms (tuned, further fragmented by
   per-subquery opt_tags in the key).

### F6. Per-iteration reorder is NOT IR-native today; a DuckDB-quality IR reorder is feasible but not a copy

- `TopDownSplitter::ReorderBeforeSplit` (src/split/topdown_splitter.cpp:397) calls
  `adapter_->ReOptimizeIR()`: a full IR→SQL→re-parse→re-optimize→IR round trip
  per iteration.
- **Measured 2026-07-06** (repeat=6, no-JIT; required first instrumenting
  `ReorderBeforeSplit`, which ran outside every timer — cost now flows into
  `extract_next_sub-IR` via `pending_extract_us_`, ir_query_splitter.cpp:948):
  topdown extract = **826 ms**, MW total 1.30 s, exe **21.9 s**; node-based
  same config: extract 455 ms, MW 537 ms, exe **8.8 s**. So the round trip is
  real (~0.8 s) but topdown's actual problem is **plan/boundary quality**
  (+13 s exe vs node-based) — the old "prime suspect" framing was wrong.
  Topdown gets no separate investment; P2 targets node-based's extract cost
  (299 ms warm on the best config — see §1 correction; 516 ms was no-cache).
- **[CORRECTED 2026-07-06 pm — the mechanism claim above was WRONG for
  node-based]** There is no per-iteration round trip in node-based: the plan
  persists as a DuckDB `LogicalOperator` across iterations. Measured
  attribution (env-guarded timers, full JOB): **~78–90% of extract cost is
  `PlanEnumerator::SolveJoinOrder` itself** — the DP enumeration + cost
  model, re-run from scratch every iteration. Split machinery, merges, and
  `ConvertPlanToIR` total ≤ 45 ms/pass. Consequence: **an IR-native port of
  the same DP does the same work and saves almost nothing.** The win must
  come from *avoiding or shrinking DP re-runs*, not relocating them.
- **Cannot copy MiddleOptimize identically**: it runs 14 passes on
  `LogicalOperator` with `Binder`, `TableFilterSet`, and storage-level HLL
  distinct stats (duckdb optimizer.cpp:469-598). SimplestIR has none of these —
  its only statistic is `estimated_cardinality` (simplest_ir.h:985).
- DP facts verified against the patched DuckDB source: exact DP ≤ 12 relations
  (`THRESHOLD_TO_SWAP_TO_APPROXIMATE`, plan_enumerator.hpp:36), greedy above
  (plan_enumerator.cpp:472-480); cost model = distinct-count ratio with
  `DEFAULT_SELECTIVITY = 0.2` (relation_statistics_helper.hpp:57 — the 0.1
  previously written here was wrong); **temp tables (LogicalColumnDataGet)
  get synthetic per-column distinct = cardinality, from_hll=false**
  (relation_statistics_helper.cpp:180-192) — the DP consumes only the temp's
  exact *cardinality*, which `UpdateRemainingIR` already injects. So the
  −1.27 s exe advantage comes from exact cardinality alone; parity does NOT
  require temp distinct counts.
- Revised P2 options (skip-if-unchanged gate / incremental DP / reduced pass
  list) — **moved to `improve_node-based.md` §3** (Track B). Quality gate
  unchanged: any scheme must match-or-beat the current re-plan orders (they
  deliver −1.27 s exe).

### F7. Runtime stats collection: move from tactical to strategic

What exists today (all tactical, at execution/codegen time):
- temp cardinality — always (the only IR stat);
- temp min/max — **on demand only**, when a kernel DIRECT_MAP build wants it
  (`GetTempTableMinMax`, duckdb_adapter.cpp:2226, ≤ 10000 distinct);
- nothing per-column otherwise (no distinct counts, no null counts).

The design: compute per-column **min/max, distinct estimate (HLL; exact when
small), null count** while materializing the temp `ColumnDataCollection` (data is
already in cache — a fused extra pass, near-free), store in a stats side-table
keyed by temp index. Consumers:
1. the IR-native reorder (F6) — needs distinct counts for the DP cost model;
2. JIT codegen — join-side choice, HT pre-sizing, DIRECT_MAP decision without
   the current on-demand rescan;
3. DuckDB itself gets **cardinality only** (Opt-7/8/12a lessons: feeding DuckDB
   more than cardinality backfired).

**[RE-SCOPED 2026-07-06 pm]** With F6 corrected, P1's value for the reorder is
near zero: DuckDB's DP never reads per-column stats for temps (it synthesizes
distinct = cardinality and ignores `statistics_extended` — verified in
relation_statistics_helper.cpp:180-192, and empirically by Opt-3: injected
min/max had *no effect* on DuckDB 1.5.2, commit 6f2ba61). Opt-3's 131 ms cost
was a **post-hoc full CDC rescan** (per-column scan loops), not a fused pass —
so the fused-pass "near-free" claim stands, but the only remaining P1
customer is (2) JIT codegen. P1 is **demoted to optional** and tracked in
`improve_node-based.md` §3. The stats side-table design stays relevant here
only as a possible input to the new splitter's boundary policy (§4.4).

### F8. The split itself is history/config-dependent — and the tuner's positional keys misbind (discovered 2026-07-06)

Evidence (same binary, same flags unless noted; group counts = subquery count
from breakdown CSV column layout / key dump):

| run | 9d | 24a | 8c |
|---|---|---|---|
| standalone, tmpl-cache (debug dump) | 1 sub + final | — | — |
| full suite, no jit-cache (release CSV) | 1 group | 1 | 1 |
| full suite, tmpl-cache fctpde (release CSV) | 5 groups | 8 | 2 |
| full suite, tmpl-cache tuned (release CSV) | 4 groups | 7 | 2 |
| full suite, tmpl-cache (debug dump, in-suite) | ~5 subs (temp2..temp5 visible) | — | — |

So node-based split decisions for the same query change with (a) suite
position (prior queries leak state) and (b) run config. Root cause not yet
identified (candidates: threshold-abort interacting with accumulated state;
storage-plan cache warmth; helper-DB state). Two consequences:

1. **Tuner key misbinding**: `tuned_per_subquery_node-based.json` keys are
   positional (query, sub-idx). For 9d it holds only `{0: interp, 1: q_tpde
   skip_off (94.9 ms)}` — derived from a 1-group split — but the tuned run
   splits 9d into 4 groups: the expensive config lands on a middle subquery
   and the final (tune key = temp_tables_.size() = 4) silently gets the
   default. Part of the current "tuned" numbers are therefore mis-tuned.
2. **P5 redesign implication**: per-subquery tuning should key on the
   **canonical plan shape** (the P4 key) instead of positional indices —
   shape keys are split-invariant and repeat-stable (F2.1), fixing the
   misbinding for free and making tune configs portable across regimes.


---

## 3. Verdicts on the 12 design points

| # | point | verdict | grounds |
|---|---|---|---|
| 1 | parallel subquery execution | **drop** (keep bg compile; tiny-side overlap = low priority) | F3 |
| 2 | cross-subquery JIT reuse | **keep** — requires shape-canonical emission + key normalization | F2 |
| 3 | stats-feedback join reorder | **keep** — the core of the new splitter | F1, F6, F7 |
| 4 | hybrid split | **keep as boundary policy, not flag-tiering** — but ceiling now measured at +0.2–0.4 s (§1); demoted to last | F4, §1 |
| 5 | richer runtime stats | **keep** — materialization-time side-table | F7 |
| 6 | IR-native reorder | **keep** — port DP core + stats, DuckDB only for iteration 0 | F6 |
| 7 | split analysis parallel to parse | **dropped** (user decision) | — |
| 8 | shape/regime-keyed tuning | **upgraded: do alongside P4** — post-fix data shows tuning wins on exe in every regime, and the optimal per-subquery config flips with the cache regime (F5.4/F5.5) | F5 |
| 9–12 | agreed as analyzed | keep per original doc | — |

---

## 4. New splitter design (feasible spec)

Working name: **stats-driven IR splitter** (SDS). All IR-level; reuses node-based
extraction machinery where possible.

Components:

1. **Startup stats snapshot** (one-time, cached like the storage plan):
   per-column distinct count (HLL), min/max, null count for all 21 IMDB base
   tables. Persisted next to `/tmp/imdb_storage_plan.cache`.

2. **Temp stats at materialization** (F7): fused collection during temp
   `ColumnDataCollection` writes; stats side-table
   `std::unordered_map<temp_index, ColumnStats[]>` owned by the splitter, not the
   IR (IR stays lean; no simplest_ir.h changes needed beyond what exists).

3. **Join-reorder cost avoidance** (F6) — **RESOLVED 2026-07-06, do NOT
   re-attempt**: Track B measured it (P2.0: expensive many-relation DP
   re-runs change the join skeleton >50% of the time → P2.a skip gate
   rejected, P2.b incremental DP skipped). The DP re-run cost (~80% of MW
   0.36 s) is accepted as the price of the −1.27 s exe advantage. What did
   land: P2.c reduced middle-pass list, default-on (−30 ms MW). The new
   splitter must KEEP the per-iteration join_order re-plan with exact temp
   cardinalities. Iteration 0 stays with the DuckDB-produced initial plan.

4. **Boundary policy** (F4): choose the split point that minimizes expected JFP
   loss: prefer boundaries whose crossing join filters are either (a) already
   enforced inside the subquery, or (b) recoverable by feeding the temp's
   min/max as an IR filter on the remainder (semi-join reduction via stats).
   Big-table-last: keep the largest base table in the final remainder so it is
   scanned once with all filters available.
   Commit the next boundary one step ahead (safe per F1: misses = 0) so split
   analysis and speculative compilation overlap execution; physical flags/plans
   stay late-bound (77.8% card_miss).

5. **Shape-canonical emission** (F2) — **DEMOTED (Track A deleted
   2026-07-06)**: measured bound only ~40–50 ms warm (all cheap small plans;
   expensive compiles canonically unique). Not part of the Phase-2 splitter.
   Revive only as P5' shape-canonical TUNE keys if tuned configs stay
   relevant (F8 misbinding fix; key design in F2/F8).

6. **Threshold/abort semantics**: keep `kSplitCardThreshold` (1M) standalone
   abort; the tune key now uses executed-subquery count (§6), so abort no longer
   corrupts per-subquery configs.

Expected wins (order of impact, RE-REVISED 2026-07-06 pm after component
attribution — Track A deleted, Track B re-scoped):
1. ~~Extract overhead~~ — **CLOSED (Track B)**: the skip-gate unknown was
   measured (P2.0: order changes >50% on expensive re-runs) → skipping
   rejected; only P2.c (−30 ms) landed. The remaining ~0.3 s extract is the
   DP enumeration that produces the −1.27 s exe advantage — a cost floor,
   not a win. Any Phase-2 splitter must match that re-plan quality.
2. Compile-time reduction from cross-query cache hits (P4) — measured bound
   **~40–50 ms warm** (F2: the shared canonical classes are all cheap small
   plans; the expensive compiles are unique). The earlier "whole jit column"
   claim was wrong (F5.6 correction). P5 re-scoped: fix the tuner's
   positional-key misbinding via shape keys (F8) — this may recover part of
   tuned's exe advantage (−0.25 s) without its compile penalty landing wrong.
3. JFP-loss reduction from boundary policy + min/max feedback — **demoted**:
   the measured ceiling is +217 ms (q_tpde) / +385 ms (q_llvm) spread over
   34/63 queries with the worst single query at ~30 ms (§1). No longer a
   headline item; bundle with P2 only if cheap.

---

## 5. Implementation plan (REVISED 2026-07-06 — dependency-ordered, incremental)

Two independent tracks; each step lands + verifies alone before the next.

**Track A — DELETED 2026-07-06** (user decision after key-dump measurement).
Measured worth: ~40–50 ms warm on the best config (74 newly-shared
compiles/pass = 12.7% of plan bytes, all cheap small plans; every expensive
compile canonically unique; cross-repeat reuse forbidden by single-run
`ClearObjCache` semantics, not by the key). Evidence preserved in F2, F5.6,
F8. Revisit only if the F8 tuner-misbinding fix ever needs a split-invariant
(shape-canonical) tune key — that key design is specified in F2/F8.

**Track B — middleware-overhead elimination: CLOSED 2026-07-06 night**
(history in `improve_node-based.md` §7–8 + `lingo-db-reuse.log`). Outcomes:
P2.0 done (skeleton changes >50% on expensive re-runs) → P2.a REJECTED;
P2.b SKIPPED; P2.c landed, later HARD-CODED (−30 ms MW; the three passes
deleted from `RunBuiltInMiddleOptimizers`, flag removed);
P1 optional (no DP consumer). MW floor reached: ~0.34 s warm, ~80% = the DP
enumeration itself, which earns its keep. Exe improvements (this document)
are the only remaining lever below ~5.26 s.

**Track C — new split strategy (THIS document)**: boundary/depth policy per
§4 + §8 (oracle). P3 (JFP-aware boundary policy, big-table-last, min/max
feedback filters, ~1k LOC) is subsumed here; the oracle result (§8) shows the
depth/threshold lever alone is worth ≥ −181 ms exe on 14 queries, well above
the +217 ms residual-loss ceiling that had demoted P3.

## 6. Open uncertainties (re-verified 2026-07-06)

1. ~~k-step boundary stability~~ — **downgraded, not blocking**: the design
   (§4.4) commits only ONE boundary ahead, which F1 verifies directly
   (misses = 0). k-step instrumentation is only needed if we later deepen the
   pipeline.
2. ~~True parallel-subquery contention~~ — moot (point dropped).
3. ~~Tuned-vs-uniform after fix~~ — RESOLVED (§7/F5): tuned wins on exe in all
   regimes; loses on total only through un-hidden compile time.
4. ~~Topdown slowness root cause~~ — **RESOLVED, hypothesis REFUTED** (measured
   2026-07-06, repeat=6 no-JIT runs; details in F6): ReOptimizeIR costs
   **826 ms** total (it previously ran OUTSIDE all timers and never reached any
   CSV column — now instrumented into `extract_next_sub-IR`,
   ir_query_splitter.cpp:948; 678/678 correct). But topdown exe = **21.9 s**
   vs node-based **8.8 s** (same config): topdown is slow because of *worse
   subquery plans/boundaries*, not the round trip. Topdown gets no separate
   investment.
5. ~~Iteration-0 order parity~~ — **moved to `improve_node-based.md` §4**
   (Track B gate).
6. ~~JFP-loss recovery magnitude~~ — **RESOLVED by measurement** (§1): ceiling
   +217 ms (q_tpde, 34 queries) / +385 ms (q_llvm, 63 queries), worst single
   query ~30 ms. P3 demoted accordingly.
7. ~~Cache-key repeat stability (incl. 28 abort queries)~~ — **RESOLVED
   empirically** (2026-07-06 key dump): 491/491 keys pairwise identical
   across iterations; two independent full-JOB dumps byte-identical.
8. ~~P4 savings magnitude~~ — **RESOLVED by measurement** (F2): 74
   newly-shared compiles/pass = 12.7% of plan bytes ≈ 40–50 ms warm; all
   expensive compiles canonically unique. Track A demoted (see §5).
9. **NEW (F8): split history/config dependence** — same binary+flags split
   9d into 1 group standalone but 4–5 groups in suite runs; root cause
   unknown (suspect: optimizer/catalog state carried across queries).
   Breaks positional tune keys and makes standalone runs non-representative.
10. ~~Skip-gate hit rate (P2.0)~~ — **moved to `improve_node-based.md` §4**.
11. ~~Attribution instrumentation~~ — **REVERTED 2026-07-06** (all
   `AQP_SPLIT_TIMING` timers and the oracle env knobs removed from both
   trees; see `improve_node-based.md` §5 and §8 below). fprintf-macro hazard
   note preserved there.

---

## 8. Oracle experiment (2026-07-06 pm): a better-than-node-based splitter is REAL

### F9. Per-query split-decision headroom measured at −11.2% exe on the heavy tail

Mechanism facts verified first (code-level):
- Node-based boundaries are **fully derivative of the join order** that
  MiddleOptimize picks each iteration: `QuerySplit::Split` →
  `TopDownSplit::VisitOperator` (duckdb query_split/top_down.cpp) assigns a
  split at every COMPARISON_JOIN's right child (pipeline-breaker rule,
  SEMI/MARK skipped); the middleware always extracts `subqueries_.front()[0]`.
  `EnumSplitAlgorithm split_algorithm = top_down` is hardcoded
  (query_split.hpp) — "node-based" = DuckDB TopDownSplit + per-iteration
  MiddleOptimize cardinality feedback.
- The only independent split decisions are the standalone-abort threshold
  (`kSplitCardThreshold` = 1M, node_based_splitter.h) and the split depth
  (how many subqueries execute before force-merging the remainder).

Setup: 14 heaviest JOB queries (16b 8c 10c 9d 17f 19d 7c 17e 18c 25c 6f 17a
6d 24a ≈ 1.61 s = 35% of node-based warm exe), best single-run config
(uniform q_tpde + tmpl-cache, repeat=15), 6 configs via two TEMPORARY env
knobs (`AQP_SPLIT_CARD_THRESHOLD`, `AQP_SPLIT_MAX_ITERS` — **reverted after
the experiment**; ~40 LOC in node_based_splitter.{h,cpp}: threshold override
+ `update_calls_` cap setting `force_terminal_next_`): base / noabort
(threshold=UINT64_MAX) / iters1 / iters2 / noabort_iters3 / thresh100k.
All 6 configs × 14 queries verified byte-identical to golden results.

Result (warm, canonical parser):

| metric | base | per-query best | Δ |
|---|---|---|---|
| exe sum | 1614.8 ms | 1433.4 ms | **−181.4 ms (−11.2%)** |
| total sum | 1693.6 ms | 1506.2 ms | **−187.4 ms** |

Per-query optimal direction is INCONSISTENT — the key finding:

| wants | queries | best deltas (ms exe) |
|---|---|---|
| deeper, uncapped abort (noabort_iters3) | 16b 17f 7c 17a | −45.5 −35.7 −27.8 −11.0 |
| fewer splits (iters1/iters2/thresh100k) | 9d 25c 18c | −41.7 −9.7 −8.2 |
| base already optimal | 10c 17e 19d 6d 6f | 0 |
| noabort | 24a | −1.5 |

Notable: 7c (the worst residual-loss query vs none-split, +31 ms) recovers
−27.8 ms with deeper splitting; 9d recovers −41.7 ms with a *stricter*
threshold. Noise floor ~1–3 ms; wins of 8–45 ms are far above it.

Uncertainties:
- Only 6 coarse configs tested → the true oracle (arbitrary per-boundary
  choices) is LARGER than −181 ms; this is a lower bound.
- Extrapolation to all 113 queries unmeasured; proportional scaling suggests
  ~−0.4–0.5 s suite-wide, but the light tail may contribute less.
- In-suite context (F8): per-query split decisions depend on suite position;
  standalone re-checks of the top winners would firm the numbers up.

Design consequence: the new splitter's core lever is a **per-query adaptive
split-depth + cardinality-threshold policy** — feedback-driven, using the
actual temp cardinalities already observed mid-loop to decide
go-deeper / stop / merge-remainder — layered on the existing node-based
machinery (§4.4 boundary policy, F4 big-table-last, F1 one-step-ahead
commit). Track B (`improve_node-based.md`) reduces the per-iteration cost of
whatever loop this policy drives.

**[SUPERSEDED IN PART 2026-07-06 late — user decision]** The policy will NOT
be layered on node-based/duckdb knobs. Phase 2 = SDS (§9): rewrite the
middleware `topdown_splitter.cpp` as a DBMS-optimizer-independent splitter;
the adaptive depth/threshold policy becomes SDS's M3. No DuckDB or
node-based changes; node-based stays the baseline and the gate (beat
exe 4.58 s).

---

## 9. SDS (stats-driven IR splitter) — DP port spec & session-verified contract (2026-07-06 late)

Decisions (user-confirmed): rewrite `src/split/topdown_splitter.cpp`;
engine contact allowed through generic SQL interfaces only (batched
EXPLAIN, one-time stats snapshot SQL); perf gate = beat node-based.

### 9.1 Architectural fact that shrinks the problem

Every sub-SQL gets DuckDB's FULL `Optimize()` at execution time on ALL jit
levels — including query-jit, which re-extracts IR from that freshly
optimized plan and compiles it (`ExecuteSQL` → `PrepareWithQueryJitAnalysis`,
duckdb_adapter.cpp:1069-1194, :2999-3045; prepared stmt = fallback).
Materialized temps give the engine exact cardinalities for free
(LogicalColumnDataGet distinct=card, relation_statistics_helper.cpp:180-192).
⇒ SDS decides **partitioning + sequencing**; intra-subquery join order is
advisory. The IR cost model only needs to pick good materialization groups.

### 9.2 Inputs (all engine-independent at decision time)

- Leaf (filtered base scan) cardinalities: already in the initial IR —
  every node carries `estimated_cardinality` (simplest_ir.h:985, populated
  duckdb_plan_to_ir.cpp:150; initial IR = DuckDB-optimized plan,
  ir_query_splitter.cpp:750-811). No EXPLAIN needed for leaves.
- Join-column distinct counts: one-time snapshot via
  `approx_count_distinct` SQL over FK/PK columns of the 21 IMDB tables,
  persisted like `/tmp/imdb_storage_plan.cache`. Cap by leaf cardinality.
- Temp tables: exact cardinality from `GetTempTableCardinality`
  (duckdb_adapter.cpp:2127-2132) via `UpdateRemainingIR`; distinct = card.
- Fallback/validation only: `BatchGetEstimatedCosts` (db_adapter.h:113,
  impl duckdb_adapter.cpp:2576-2609, `explain_cache_`d; one
  parse+optimize per uncached SQL — use sparingly).

### 9.3 DP port surface (from duckdb src/optimizer/join_order/)

- Enumeration: DPccp — per start node `EmitCSG` +
  `EnumerateCSGRecursive` with exclusion sets; neighbor subsets via
  `GetAllNeighborSets` (plan_enumerator.cpp:46-76); `TryEmitPair` soft
  timeout at 10K pairs → greedy (plan_enumerator.cpp:168-182). Exact DP for
  ≤ 12 relations (`THRESHOLD_TO_SWAP_TO_APPROXIMATE`, plan_enumerator.hpp:36);
  greedy `SolveJoinOrderApproximately` above (:341-444): repeatedly join
  min-cost connected pair, cross-product two smallest if disconnected.
- Cost: `cost = join_card + left_cost + right_cost` (cost_model.cpp:13-18).
- Cardinality: `prod(leaf cards in set) / denominator`; denominator built
  from equality-edge equivalence sets (tdoms) in descending-tdom order;
  tdom = max(hll, no-hll) distinct; non-equi edge → tdom^(2/3)
  (cardinality_estimator.cpp:410-426, :256); SEMI/ANTI: RHS excluded from
  numerator, ×1/5 (DEFAULT_SEMI_ANTI_SELECTIVITY, cardinality_estimator.hpp:92).
- Filters (only if we ever re-derive leaf cards ourselves):
  equality-on-stats reduces card; else non-optional filters →
  DEFAULT_SELECTIVITY 0.2 (relation_statistics_helper.hpp:57).
- Graph edges from EQUALITY predicates only; non-equi kept as filters,
  not edges (query_graph_manager.cpp:79-150). Reorderable relations:
  scans/chunks/blocking ops; INNER/SEMI/ANTI comparison joins reorderable,
  LEFT/RIGHT/FULL blocking (relation_manager.cpp:96-163). Bushy allowed.
- NOT ported (engine-only, not needed): Binder, TableFilterSet, HLL
  statistics callbacks, the 14 MiddleOptimize rewrite passes (the engine
  re-runs rewrites per sub-SQL anyway, §9.1).

### 9.4 Split-loop contract the rewrite must honor

Loop `ExecuteSplitLoop` ir_query_splitter.cpp:850: Preprocess :888 →
[per iter] ReorderBeforeSplit :954 (timed into `pending_extract_us_`
:949-956) → SplitIR :2032 → ExecuteSubIR :2759 (GenerateSQL →
ExecuteSQLandCreateTempTable → GetTempTableCardinality) → UpdateRemainingIR
:2691 → generic UpdateRemainingIRIndices unless `SkipUpdateIndices()`.
`SubqueryExtraction` (split_algorithm.h:21-69): `executed_table_indices`,
`sub_ir` (owned) OR `pipeline_breaker_ptr` (subtree in remaining IR),
`temp_table_name` ("tempN", :2783), `is_final` (sets remaining_ir=sub_ir,
stops loop). column_mappings from executable IR target_list (:2664-2671),
aliases "{table}_{col}". Debug-only CROSS_PRODUCT abort on executable IR
(:2086-2108) — SDS must not emit cross-product subqueries (disconnected
graph → keep components whole / cross-product only in final remainder,
which the engine handles).

### 9.5 Milestones

M1 skeleton (DP module `ir_join_optimizer.{h,cpp}` + splitter rewrite +
distinct snapshot; partition rule mirrors node-based granularity — inspect
8c/9d sub-SQLs first). M2 feedback loop + boundary policy (big-table-last,
JFP-aware, monotone-reduction check); gate exe within 10% of node-based →
parity. M3 adaptive depth/threshold (F9); verify est-vs-actual error signal
separates the oracle's "deeper" vs "fewer" groups BEFORE building the rule;
gate: beat 4.58 s. M4 optional: F7 temp stats, min/max cross-boundary
filters, P5' shape-keyed tuning.

Open questions → M1: node-based materialization granularity; F9 transfer
to SDS boundaries. Risk fallback: batched-EXPLAIN validation of candidate
partitions per iteration if the IR DP misjudges many-relation queries.
