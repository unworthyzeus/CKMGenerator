#!/bin/bash
# Try 80 cleanup on sert-2001: 1 GPU only (avoids MaxGRESPerAccount from chained pending jobs).
# Targets ONLY the processes of the previous train job (via PREV_JOB_ID).
#SBATCH --job-name=ckm-t80-cleanup-sert2001
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-00:05:00
#SBATCH --output=logs_cleanup_try80_%j.out
#SBATCH --error=logs_cleanup_try80_%j.err

set -euo pipefail

echo "[cleanup] host=$(hostname) start=$(date --iso-8601=seconds) prev_job=${PREV_JOB_ID:-<unset>}"
nvidia-smi || true

if [ -n "${PREV_JOB_ID:-}" ]; then
  # scontrol listpids <jobid> returns only PIDs that belong to that specific
  # job's cgroup on this node — bounded kill, never touches other chains.
  pids=$(scontrol listpids "$PREV_JOB_ID" 2>/dev/null | awk 'NR>1 && $1 ~ /^[0-9]+$/ {print $1}' | sort -u)
  if [ -n "${pids:-}" ]; then
    echo "[cleanup] killing leftover pids from job $PREV_JOB_ID: $pids"
    for pid in $pids; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in $pids; do
      kill -KILL "$pid" 2>/dev/null || true
    done
  else
    echo "[cleanup] no residual pids for job $PREV_JOB_ID (normal)"
  fi
else
  echo "[cleanup] PREV_JOB_ID not set — skipping process kill"
fi

echo "[cleanup] nvidia-smi after"
nvidia-smi || true
echo "[cleanup] end=$(date --iso-8601=seconds)"
