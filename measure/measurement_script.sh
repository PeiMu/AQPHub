# none jit
bash ./hyperfine_aqp.sh job duckdb none none o1 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb relationship-center none o1 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb node-based none o1 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb topdown none o1 none off off off off off off off && \
# expr jit none split
bash ./hyperfine_aqp.sh job duckdb none expr o1 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb none expr o2 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb none expr o3 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb none expr o1 auto off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb none expr o2 auto off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb none expr o3 auto off off off off off off off && \
# expr jit relationship-center
bash ./hyperfine_aqp.sh job duckdb relationship-center expr o1 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb relationship-center expr o2 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb relationship-center expr o3 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb relationship-center expr o1 auto off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb relationship-center expr o2 auto off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb relationship-center expr o3 auto off off off off off off off && \
# expr jit node-based
bash ./hyperfine_aqp.sh job duckdb node-based expr o1 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb node-based expr o2 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb node-based expr o3 none off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb node-based expr o1 auto off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb node-based expr o2 auto off off off off off off off && \
bash ./hyperfine_aqp.sh job duckdb node-based expr o3 auto off off off off off off off && \
# expr jit topdown
bash ./hyperfine_aqp.sh job duckdb topdown expr o1 none off off off off off off off && \
#bash ./hyperfine_aqp.sh job duckdb topdown expr o2 none off off off off off off off && \
#bash ./hyperfine_aqp.sh job duckdb topdown expr o3 none off off off off off off off && \
#bash ./hyperfine_aqp.sh job duckdb topdown expr o1 auto off off off off off off off && \
#bash ./hyperfine_aqp.sh job duckdb topdown expr o2 auto off off off off off off off && \
#bash ./hyperfine_aqp.sh job duckdb topdown expr o3 auto off off off off off off off && \
# operator jit none split
bash ./hyperfine_aqp.sh job duckdb none operator o1 none && \
bash ./hyperfine_aqp.sh job duckdb none operator o2 none && \
bash ./hyperfine_aqp.sh job duckdb none operator o3 none && \
bash ./hyperfine_aqp.sh job duckdb none operator o1 auto && \
bash ./hyperfine_aqp.sh job duckdb none operator o2 auto && \
bash ./hyperfine_aqp.sh job duckdb none operator o3 auto 
