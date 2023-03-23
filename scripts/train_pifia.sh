#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=t4v2
#SBATCH --qos=normal
#SBATCH --output=./pifia_slurm-%A_%a.out
#SBATCH --error=./pifia_slurm-%A_%a.err
#SBATCH --open-mode=append

export HDD_MODELS_DIR=./ # change this dir to where you want to save the actual model weights (e.g. HDD drive)

CHECKPOINTS=(
  "PIFIA_toy_dataset"
)

# set up checkpointing
ckpt=$PWD/ckpt_dir
log_file=$PWD/log_file.log

ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $ckpt
touch $ckpt/DELAYPURGE

# export PATH=/pkgs/anaconda3/bin:$PATH
# export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pkgs/cudnn-10.1-v7.6.3.30/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pkgs/TensorRT-6/lib:$LD_LIBRARY_PATH
# export PATH=/h/anastasia/anaconda3/bin:$PATH
source activate conda_env

cmd="python model/train.py --dataset harsha ${HPARAMS[SLURM_ARRAY_TASK_ID]} \
    --backbone pifia_network --learning_rate 0.0003 --dropout_rate 0.02 --cosine_decay True \
    --labels_type toy_dataset --dense1_size 128 --num_features 64 --save_prefix TEST_RUN
    --num_epoch 30  --checkpoint_interval 1800 --checkpoint_dir $ckpt --log_file $log_file"

echo $cmd
eval $cmd
