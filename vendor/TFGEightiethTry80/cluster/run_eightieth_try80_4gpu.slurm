#!/bin/bash
#SBATCH --job-name=ckm-t80-4gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=logs_train_try80_%x_%j.out
#SBATCH --error=logs_train_try80_%x_%j.err

export NPROC_PER_NODE=4
source cluster/run_eightieth_try80_1gpu.slurm
