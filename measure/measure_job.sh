#!/bin/bash

mkdir -p job_result/
rm -rf compile.log

engine=$1
split=$2

########################################
# Start / Stop PostgreSQL
########################################
Project_path=/home/pei/Project/project_bins
pg_start() {
  pg_ctl start -l $Project_path/logfile -D $Project_path/data
}
pg_stop() {
  pg_ctl stop -D $Project_path/data -m smart -s
}
rm_pg_log() {
  rm $Project_path/logfile
}

########################################
# Start / Stop Umbra
########################################
container_name="umbra_benchmark"
start_umbra() {
    echo "Starting Umbra docker..."

    docker run -d \
        --name "$container_name" \
        --network=host \
	--cpuset-cpus="0" \
        -e OMP_NUM_THREADS=1 \
        -e OMP_THREAD_LIMIT=1 \
        -e OMP_PROC_BIND=TRUE \
        -v umbra-db:/var/db \
        -v /tmp:/tmp \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 --port 15432 /var/db/imdb.db >/dev/null

    wait_for_umbra
}

stop_umbra() {
    echo "Stopping Umbra docker..."
    docker stop "$container_name" >/dev/null 2>&1 || true
    docker rm "$container_name" >/dev/null 2>&1 || true
}

wait_for_umbra() {
    echo "Waiting for Umbra to accept connections on port 15432..."
    until pg_isready -h localhost -p 15432 >/dev/null 2>&1; do
        sleep 1
    done
    echo "Umbra is ready."
}

########################################
# Start / Stop MariaDB
########################################
mariadb_start() {
    sudo systemctl start mariadb
}

mariadb_stop() {
    sudo systemctl stop mariadb
}


########################################
# Start / Stop OpenGauss
########################################
opengauss_start() {
    sudo systemctl start opengauss
}

opengauss_stop() {
    sudo systemctl stop opengauss
}

cleanup() {
    if [[ "$engine" == "umbra" ]]; then
        stop_umbra
    elif [[ "$engine" == "mariadb" ]]; then
        mariadb_stop
        pg_stop
    elif [[ "$engine" == "postgres" ]]; then
        pg_stop
    elif [[ "$engine" == "opengauss" ]]; then
        opengauss_stop
    fi
}
trap cleanup EXIT


#echo "compiling..."
#bash ./compile.sh >> compile.log 2>&1
#echo "compilation done"

########################################
# Start Engine
########################################
if [[ "$engine" == "umbra" ]]; then
    start_umbra
elif [[ "$engine" == "mariadb" ]]; then
    mariadb_start
    pg_start
elif [[ "$engine" == "postgres" ]]; then
    pg_start
elif [[ "$engine" == "opengauss" ]]; then
    opengauss_start
fi

########################################
# ANALYZE
########################################
echo "ANALYZING..."
if [[ "$engine" == "umbra" ]]; then
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -c "ANALYZE;"
elif [[ "$engine" == "mariadb" ]]; then
    mariadb -u imdb -D imdb < /home/pei/Project/benchmarks/imdb_job-postgres/analyze_mariadb_table.sql
elif [[ "$engine" == "postgres" ]]; then
    psql -U pei -d imdb -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 -c "ANALYZE;"
fi
echo "ANALYZE done"

cd ../measure && bash ./hyperfine_job.sh "${engine}" "${split}"

#mv compile.log job_result/.
