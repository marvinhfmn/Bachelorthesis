#!/bin/bash -l
#
#SBATCH --job-name=GraphNetTraining
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx3080:2
#SBATCH --partition rtx3080
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_submit_outputs/%x_%j.out  # Redirect standard output
#SBATCH --error=/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_submit_outputs/%x_%j.err   # Redirect standard error
#SBATCH --export=NONE

# Source the bash configuration to ensure conda is initialized
source ~/.bashrc

unset SLURM_EXPORT_ENV

# module load python
conda activate graphnet_cuda

# debugging flags 
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo $TMPDIR

# Measure the time taken for the cp command
start_time=$(date +%s)
cp -r "/home/wecapstor3/capn/capn108h/selection_databases/allflavor_classif8_9_19_20_22_23_26_27" "$TMPDIR"
end_time=$(date +%s)
cp_duration=$((end_time - start_time))
echo "Time taken for cp command: $cp_duration seconds"

# List all files in TMPDIR
echo "Files in TMPDIR after cp command:"
ls -R "$TMPDIR"

# run script
# srun ~/miniconda3/envs/graphnet_cuda/bin/python /home/saturn/capn/capn108h/programming_GNNs_and_training/GNN.py --config /home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml 

# srun python3 /home/saturn/capn/capn108h/programming_GNNs_and_training/GNN.py --config /home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml  --use_tmpdir True 

srun python3 /home/saturn/capn/capn108h/programming_GNNs_and_training/GNN.py --config /home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml --resumefromckpt True --use_tmpdir True 