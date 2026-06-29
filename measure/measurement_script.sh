# none jit
bash ./measure_job.sh duckdb none none o1 none off off off off off off off && \
bash ./measure_job.sh duckdb relationship-center none o1 none off off off off off off off && \
bash ./measure_job.sh duckdb node-based none o1 none off off off off off off off && \
bash ./measure_job.sh duckdb topdown none o1 none off off off off off off off && \
# expr jit none split
bash ./measure_job.sh duckdb none expr o1 none off off off off off off off && \
bash ./measure_job.sh duckdb none expr o2 none off off off off off off off && \
bash ./measure_job.sh duckdb none expr o3 none off off off off off off off && \
bash ./measure_job.sh duckdb none expr o1 auto off off off off off off off && \
bash ./measure_job.sh duckdb none expr o2 auto off off off off off off off && \
bash ./measure_job.sh duckdb none expr o3 auto off off off off off off off && \
# expr jit relationship-center
bash ./measure_job.sh duckdb relationship-center expr o1 none off off off off off off off && \
bash ./measure_job.sh duckdb relationship-center expr o2 none off off off off off off off && \
bash ./measure_job.sh duckdb relationship-center expr o3 none off off off off off off off && \
bash ./measure_job.sh duckdb relationship-center expr o1 auto off off off off off off off && \
bash ./measure_job.sh duckdb relationship-center expr o2 auto off off off off off off off && \
bash ./measure_job.sh duckdb relationship-center expr o3 auto off off off off off off off && \
# expr jit node-based
bash ./measure_job.sh duckdb node-based expr o1 none off off off off off off off && \
bash ./measure_job.sh duckdb node-based expr o2 none off off off off off off off && \
bash ./measure_job.sh duckdb node-based expr o3 none off off off off off off off && \
bash ./measure_job.sh duckdb node-based expr o1 auto off off off off off off off && \
bash ./measure_job.sh duckdb node-based expr o2 auto off off off off off off off && \
bash ./measure_job.sh duckdb node-based expr o3 auto off off off off off off off && \
# expr jit topdown
bash ./measure_job.sh duckdb topdown expr o1 none off off off off off off off && \
#bash ./measure_job.sh duckdb topdown expr o2 none off off off off off off off && \
#bash ./measure_job.sh duckdb topdown expr o3 none off off off off off off off && \
#bash ./measure_job.sh duckdb topdown expr o1 auto off off off off off off off && \
#bash ./measure_job.sh duckdb topdown expr o2 auto off off off off off off off && \
#bash ./measure_job.sh duckdb topdown expr o3 auto off off off off off off off && \
# operator jit none split
bash ./measure_job.sh duckdb none operator o1 none && \
bash ./measure_job.sh duckdb none operator o2 none && \
bash ./measure_job.sh duckdb none operator o3 none && \
bash ./measure_job.sh duckdb none operator o1 auto && \
bash ./measure_job.sh duckdb none operator o2 auto && \
bash ./measure_job.sh duckdb none operator o3 auto 
