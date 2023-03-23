#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --partition=t4v2
#SBATCH --qos=normal
#SBATCH --output=./features_slurm-%A_%a.out
#SBATCH --error=./features_slurm-%A_%a.err
#SBATCH --open-mode=append

export HDD_MODELS_DIR=./ # change this dir to where you want to save the actual model weights (e.g. HDD drive)

CHECKPOINTS=(
  "1"
  "2"
  "3"
  "4"
  "5"
  "6"
  "7"
  "8"
  "9"
  "10"
)

source activate conda_env

HPARAMS=(
   "--epoch 1"
   "--epoch 2"
   "--epoch 3"
   "--epoch 4"
   "--epoch 5"
   "--epoch 6"
   "--epoch 7"
   "--epoch 8"
   "--epoch 9"
   "--epoch 10"
)

cmd="python model/extract_features.py --dataset harsha --labels_type toy_dataset \
     --backbone pifia_network ${HPARAMS[SLURM_ARRAY_TASK_ID]} --model_dir ./saved_weights/ \
     --num_features 64 --dense1_size 128 --layer d2 --log_file ./log_file.log"

echo $cmd
eval $cmd