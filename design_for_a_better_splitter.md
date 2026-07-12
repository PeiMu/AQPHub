We found 16 queries where node-based execution is worse than none-split, costing 312ms total for no-jit. Root cause: the splitter produces a join order where intermediate cardinality grows rather than shrinks across sub-queries (e.g., 8c: 11ms → 52ms → 68ms → 116ms)

Here are some points for brainstorm:
To design a new split strategy (still based on our IR level, in third_party/IR_SQL_Converter/inc/simplest_ir.h), consider: 
1. can we fully use multi-thread for running different no-dependency sub-plans in parallel? Will it conflict to spec-jit?
2. can our new split strategy create subqueries that the jit compiled binary/object can be reuse across different subqueries? Or is it possbile to have incremental jit compilation for the later subqueries to reuse what we have compiled?
3. can we still split the original plan into subplans once at the beginning so that we can do JIT compilation in pipeline-flow for different subplans to hide the compilation latency. But do we need to incremental compile, or fall back to DuckDB interpretation if the later subplans changed due to join reorder from updated runtime info (like the current spec-jit)? 
4. One idea of hybrid split strategy is: for the first few subplans, we execute them by native DuckDB interpretation or light jit (e.g., expr-jit or operator-jit), so we split as the node-based way; for the last few subplans, we have opportunity to hide heavy JIT compilation (or even csr building latency with kernel-path, if it is possible to build the csr, which is usually large, while the first few subquery execution), and will run them in pipeline-jit or query-jit (or even the kernel path), so we can split in another way. 
5. For either DuckDB's join order optimization or our JIT optimization, what runtime info can be helpful from subplan execution? Check /home/pei/Project/BespokeOLAP/conversations/prompts/expert_knowledge.txt, /home/pei/Project/GenDB/src/gendb/agents/code-generator/prompt.md, and /home/pei/Project/GenDB/src/gendb/agents/query-optimizer/prompt.md. 
6. for the new split strategy, do we still need the pre-optimizer from duckdb? analyze what benefits or negative of this.
7. For AQP_middleware system, while DuckDB parse and preprocess, can we do quick analysis for split strategy in parallel to hide this latency, if there is any? 
8. can the "per-query flag tuning" idea still work with the new split strategy?
9. Check through the source code, is there any other effect that the new split strategy can help but I missed?
10. The golden result is measure/duckdb_job_no-split_golden.txt
11. Is it possible to implement the new split strategy based on include/split/topdown_splitter.h, or we need to build from scratch?
12. check through the expr-jit, operator-jit, pipeline-jit, and query-jit with all jit-related flags, list all points what is the new split strategy design should be.
