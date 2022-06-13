#!/bin/sh
export COMPSS_PYTHON_VERSION=None
#module load dislib/0.6.4-COMPSs_2.9-qr
#module load dislib/0.7.0
module load dislib/master
module load COMPSs/2.10
module load python/3.7.4
export PYTHONPATH=$PWD:$PYTHONPATH

export ComputingUnits=8

enqueue_compss  --qos=debug \
	--log_level=info \
	--pythonpath=/gpfs/scratch/bsc08/bsc08293/AI-SPRINT \
	--base_log_dir=/gpfs/scratch/bsc08/bsc08293 \
	--num_nodes=2  \
	--exec_time=120  \
	--python_interpreter=python3 \
/gpfs/scratch/bsc08/bsc08293/AI-SPRINT/train_csvm_dislib.py good_model pickle \
/gpfs/scratch/bsc08/bsc08293/AI-SPRINT/data/training2017/ 200 200 200

