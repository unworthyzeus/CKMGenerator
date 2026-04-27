#!/bin/bash
#SBATCH --job-name=ckm-t80-2gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:2
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=24
#SBATCH --mem=72G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=logs_train_try80_%x_%j.out
#SBATCH --error=logs_train_try80_%x_%j.err

export NPROC_PER_NODE=2
source cluster/run_eightieth_try80_1gpu.slurm
