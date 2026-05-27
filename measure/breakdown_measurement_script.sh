# none jit
bash ./measure_breakdown_time_job.sh duckdb none none o1 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center none o1 none off off off off off off off none && \
bash ./measure_breakdown_time_job.sh duckdb node-based none o1 none off off off off off off off none && \
## expr jit none split
#bash ./measure_breakdown_time_job.sh duckdb none expr o1 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none expr o2 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none expr o3 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none expr o1 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none expr o2 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none expr o3 auto off off off off off off off none && \
## expr jit relationship-center
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr o1 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr o2 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr o3 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr o1 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr o2 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr o3 auto off off off off off off off none && \
## expr jit node-based
#bash ./measure_breakdown_time_job.sh duckdb node-based expr o1 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based expr o2 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based expr o3 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based expr o1 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based expr o2 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based expr o3 auto off off off off off off off none && \
## operator jit none split
#bash ./measure_breakdown_time_job.sh duckdb none operator o1 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none operator o2 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none operator o3 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none operator o1 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none operator o2 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb none operator o3 auto off off off off off off off none && \
## operator jit relationship-center
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator o1 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator o2 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator o3 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator o1 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator o2 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator o3 auto off off off off off off off none && \
## operator jit node-based
#bash ./measure_breakdown_time_job.sh duckdb node-based operator o1 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based operator o2 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based operator o3 none off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based operator o1 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based operator o2 auto off off off off off off off none && \
#bash ./measure_breakdown_time_job.sh duckdb node-based operator o3 auto off off off off off off off none && \
## pipeline jit none split (CSR has no effect on none-split, use none)
#bash ./measure_breakdown_time_job.sh duckdb none pipeline o1 none on on on on on on off none && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline o2 none on on on on on on off none && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline o3 none on on on on on on off none && \
bash ./measure_breakdown_time_job.sh duckdb none pipeline o1 auto on on on on on on off none && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline o2 auto on on on on on on off none && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline o3 auto on on on on on on off none && \
## pipeline jit relationship-center
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline o1 none on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline o2 none on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline o3 none on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline o1 auto on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline o2 auto on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline o3 auto on on on on on on off inner && \
## pipeline jit node-based
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline o1 none on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline o2 none on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline o3 none on on on on on on off inner && \
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline o1 auto on on on on on on off inner
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline o2 auto on on on on on on off inner && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline o3 auto on on on on on on off inner
