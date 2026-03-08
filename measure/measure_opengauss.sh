#!/bin/bash

mkdir -p job_result/
rm -rf compile.log

log_name=opengauss_official.csv
rm -rf ${log_name}
rm -rf temp.csv

########################################
# Start / Stop opengauss
########################################
opengauss_start() {
    sudo systemctl start opengauss
}

opengauss_stop() {
    sudo systemctl stop opengauss
}

cleanup() {
    opengauss_stop
}
trap cleanup EXIT

opengauss_start

########################################
# ANALYZE
########################################
echo "ANALYZING..."
sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 < /home/pei/Project/benchmarks/imdb_job-postgres/analyze_table.sql
echo "ANALYZE done"

#dir="$JOB_PATH/opengauss_queries"
dir="$JOB_PATH/queries"
iteration=10

for sql in "${dir}"/*.sql; do
  #echo "hyperfine run ${sql}" 2>&1|tee -a ${log_name}
  hyperfine --warmup 5 --runs ${iteration} --export-csv temp.csv "sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 < ${sql}"
  cat temp.csv >> ${log_name}
done

mv ${log_name} job_result/.
rm -rf temp.csv

