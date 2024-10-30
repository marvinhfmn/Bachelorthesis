#!/bin/bash -l
#
#SBATCH --job-name=GNNprediction
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition a100
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_purepredict_outputs/%x_%j.out  
#SBATCH --error=/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_purepredict_outputs/%x_%j.out   # Redirect both stdout and stderr
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
find "$SRC_DIR" -type f -name "*test*" -print | xargs -i -P "$NUM_PROCS" cp {} "$TMPDIR/"
# cp -r "/home/wecapstor3/capn/capn108h/selection_databases/allflavor_classif8_9_19_20_22_23_26_27" "$TMPDIR"
end_time=$(date +%s)
cp_duration=$((end_time - start_time))
echo "Time taken for cp command: $cp_duration seconds"

# List all files in TMPDIR
echo "Files in TMPDIR after cp command:"
ls -R "$TMPDIR"

subfolder="/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_EnergyReco"
# ckpt_path="/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_EnergyReco/checkpoints/DynEdge-epoch=5-val_loss=0.12-train_loss=0.12.ckpt"
ckpt_path="/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_EnergyReco/checkpoints/DynEdge-epoch=41-val_loss=0.09-train_loss=0.08.ckpt"
pth_path="/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_EnergyReco/GNN_DynEdge_mergedNuE_NuMu_NuTau.pth"
config="/home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml"

# run script
srun python3 /home/saturn/capn/capn108h/programming_GNNs_and_training/evaluateGNN.py --config "$config" --use_tmpdir --ckpt_or_statedict_path "$ckpt_path" --subfolder "$subfolder" --no_plotting
# srun python3 /home/saturn/capn/capn108h/programming_GNNs_and_training/evaluateGNN.py --config "$config" --use_tmpdir --ckpt_or_statedict_path "$pth_path" --subfolder "$subfolder"
