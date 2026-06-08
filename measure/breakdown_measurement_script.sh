# none jit
bash ./measure_breakdown_time_job.sh duckdb none none none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb relationship-center none none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb node-based none none off off off off off off off && \
# expr jit none split
bash ./measure_breakdown_time_job.sh duckdb none expr none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb none expr auto off off off off off off off && \
# expr jit relationship-center
bash ./measure_breakdown_time_job.sh duckdb relationship-center expr none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb relationship-center expr auto off off off off off off off && \
# expr jit node-based
bash ./measure_breakdown_time_job.sh duckdb node-based expr none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr auto off off off off off off off && \
# operator jit none split
bash ./measure_breakdown_time_job.sh duckdb none operator none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb none operator auto off off off off off off off && \
# operator jit relationship-center
bash ./measure_breakdown_time_job.sh duckdb relationship-center operator none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb relationship-center operator auto off off off off off off off && \
# operator jit node-based
bash ./measure_breakdown_time_job.sh duckdb node-based operator none off off off off off off off && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator auto off off off off off off off && \
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
bash ./measure_breakdown_time_job.sh lingodb none none none off off off off off off off && \
bash ./measure_breakdown_time_job.sh lingodb relationship-center none none off off off off off off off && \
bash ./measure_breakdown_time_job.sh lingodb node-based none none off off off off off off off
