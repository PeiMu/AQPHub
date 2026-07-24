#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

mkdir -p job_result/
rm -rf compile.log

log_name=opengauss_official_result.txt
rm -rf ${log_name}

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
sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 < "${JOB_PATH}/analyze_table.sql"
echo "ANALYZE done"

#dir="$JOB_PATH/opengauss_queries"
dir="$JOB_PATH/queries"

for sql in "${dir}"/*.sql; do
  echo "Running benchmark for $sql..." | tee -a "$log_name"
  sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 < ${sql} 2>&1 | tee -a "$log_name"
done

mv ${log_name} job_result/.

