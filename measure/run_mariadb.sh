#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

mkdir -p job_result/
rm -rf compile.log

log_name=mariadb_official_result.txt
rm -rf ${log_name}

########################################
# Start / Stop MariaDB
########################################
mariadb_start() {
    sudo systemctl start mariadb
}

mariadb_stop() {
    sudo systemctl stop mariadb
}

#cleanup() {
#    mariadb_stop
#}
#trap cleanup EXIT

#mariadb_start

########################################
# ANALYZE
########################################
echo "ANALYZING..."
mariadb -u imdb -D imdb < "${JOB_PATH}/analyze_mariadb_table.sql"
echo "ANALYZE done"

#dir="$JOB_PATH/mariadb_queries"
dir="$JOB_PATH/queries"

for sql in "${dir}"/*.sql; do
  echo "Running benchmark for $sql..." | tee -a "$log_name"
  mariadb -u imdb -D imdb < ${sql} 2>&1 | tee -a "$log_name"
done

mv ${log_name} job_result/.

