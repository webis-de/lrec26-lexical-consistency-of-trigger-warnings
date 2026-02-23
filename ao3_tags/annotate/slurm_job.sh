#!/bin/bash
#SBATCH --job-name=ao3-llm-annotations
#SBATCH --gres=gpu:ampere
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=log-%j.log
#SBATCH --container-image=./hf-transformers-pytorch.sqsh
#SBATCH --container-mounts=/mnt/ceph:/mnt/ceph
#SBATCH --time=1440

warning=$1
category=$2
job_id=$3
python3 -m ao3_tags.annotate.llm \
          ${warning} \
          ${category} \
          ${job_id} \
          --n_profiles=10 \
          --batch_size=16 \
          --model_name="google/flan-t5-xxl" \
          --sample_passages=$4 \
          --num_passages=$5