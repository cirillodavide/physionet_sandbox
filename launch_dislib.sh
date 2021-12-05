#!/bin/sh
export COMPSS_PYTHON_VERSION=None
#module load dislib/0.6.4-COMPSs_2.9-qr
module load dislib/0.7.0
module load COMPSs/2.10
module load python/3.7.4
export PYTHONPATH=$PWD:$PYTHONPATH

export ComputingUnits=8

worker_working_dir=/gpfs/scratch/bsc19/bsc08293/PhysioNet
base_log_dir=/gpfs/scratch/bsc08/bsc08293/

queue=bsc_ls
#queue=debug
time_limit=5*60
num_nodes=2 #4

# log level off for better performance
enqueue_compss --qos=$queue \
 --log_level=off \
 --job_name=csvm_small \
 --worker_in_master_cpus=0 \
 --jvm_master_opts="-Xms16000m,-Xmx50000m,-Xmn1600m" \
 --max_tasks_per_node=48 \
 --exec_time=$time_limit \
 --num_nodes=$num_nodes \
 --base_log_dir=${base_log_dir} \
 --worker_working_dir=${worker_working_dir} \
 davide.py
