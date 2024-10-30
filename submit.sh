#!/bin/bash -l
#
#SBATCH --job-name=EnergyReco
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx3080:6
#SBATCH --partition rtx3080
#SBATCH --ntasks-per-node=6
#SBATCH --time=24:00:00
#SBATCH --output=/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_submit_outputs/%x_%j.out  
#SBATCH --error=/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_submit_outputs/%x_%j.out   # Redirect both stdout and stderr
#SBATCH --export=NONE

# Source the bash configuration to ensure conda is initialized
source ~/.bashrc

unset SLURM_EXPORT_ENV

conda activate graphnet_cuda

# debugging flags 
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# echo $TMPDIR

SRC_DIR="/home/woody/capn/$USER"
NUM_PROCS=4

# Measure the time taken for the cp command
start_time=$(date +%s)
find "$SRC_DIR" -type f -print | xargs -i -P "$NUM_PROCS" cp {} "$TMPDIR/"
# cp -r "/home/wecapstor3/capn/capn108h/selection_databases/allflavor_classif8_9_19_20_22_23_26_27" "$TMPDIR"
end_time=$(date +%s)
cp_duration=$((end_time - start_time))
echo "Time taken for cp command: $cp_duration seconds"

# List all files in TMPDIR
echo "Files in TMPDIR after cp command:"
ls -R "$TMPDIR"

# run script
# srun ~/miniconda3/envs/graphnet_cuda/bin/python /home/saturn/capn/capn108h/programming_GNNs_and_training/GNN.py --config /home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml 

# srun python3 /home/saturn/capn/capn108h/programming_GNNs_and_training/GNN.py --config /home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml  --use_tmpdir  

srun python3 /home/saturn/capn/capn108h/programming_GNNs_and_training/GNN.py --config /home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml --resumefromckpt --use_tmpdir 

# SECONDS=0
# while [ "$SECONDS" -le 86400 ]; do
#     sleep 60
# done

# cd ${SLURM_SUBMIT_DIR}
# sbatch $0