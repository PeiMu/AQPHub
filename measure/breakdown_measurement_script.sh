# none jit
bash ./measure_breakdown_time_job.sh duckdb none none && \
bash ./measure_breakdown_time_job.sh duckdb relationship-center none && \
bash ./measure_breakdown_time_job.sh duckdb node-based none && \
# expr jit none split
bash ./measure_breakdown_time_job.sh duckdb none expr none && \
bash ./measure_breakdown_time_job.sh duckdb none expr auto && \
# expr jit relationship-center
bash ./measure_breakdown_time_job.sh duckdb relationship-center expr none && \
bash ./measure_breakdown_time_job.sh duckdb relationship-center expr auto && \
# expr jit node-based
bash ./measure_breakdown_time_job.sh duckdb node-based expr none && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr auto && \
# operator jit none split
bash ./measure_breakdown_time_job.sh duckdb none operator none && \
bash ./measure_breakdown_time_job.sh duckdb none operator auto && \
# operator jit relationship-center
bash ./measure_breakdown_time_job.sh duckdb relationship-center operator none && \
bash ./measure_breakdown_time_job.sh duckdb relationship-center operator auto && \
# operator jit node-based
bash ./measure_breakdown_time_job.sh duckdb node-based operator none && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator auto && \
# pipeline jit none split
bash ./measure_breakdown_time_job.sh duckdb none pipeline none && \
bash ./measure_breakdown_time_job.sh duckdb none pipeline auto && \
# pipeline jit relationship-center
bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline none && \
bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline auto && \
# pipeline jit node-based
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none && \
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline auto && \
# pipeline kernel none split
bash ./measure_breakdown_time_job_kernel.sh duckdb none pipeline none && \
bash ./measure_breakdown_time_job_kernel.sh duckdb none pipeline auto && \
# pipeline kernel relationship-center
bash ./measure_breakdown_time_job_kernel.sh duckdb relationship-center pipeline none && \
bash ./measure_breakdown_time_job_kernel.sh duckdb relationship-center pipeline auto && \
# pipeline kernel node-based
bash ./measure_breakdown_time_job_kernel.sh duckdb node-based pipeline none && \
bash ./measure_breakdown_time_job_kernel.sh duckdb node-based pipeline auto && \
# lingodb (no JIT support — LingoDB uses its own compiler pipeline)
bash ./measure_breakdown_time_job.sh lingodb none none && \
bash ./measure_breakdown_time_job.sh lingodb relationship-center none && \
bash ./measure_breakdown_time_job.sh lingodb node-based none 
