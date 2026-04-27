#!/bin/bash
# Try 80 cleanup on sert-2001.
# First try the previous job's cgroup; if timed-out/cancelled workers escaped
# it, fall back to Try-80-specific CUDA/process matching for this user.
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

TARGET_USER="${TARGET_USER:-$USER}"
declare -A PIDS_TO_KILL=()

log() {
  echo "[cleanup] $*"
}

add_pid() {
  local pid="$1"
  if [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" -gt 1 ] && [ "$pid" -ne "$$" ]; then
    PIDS_TO_KILL["$pid"]=1
  fi
}

collect_prev_job_pids() {
  if [ -z "${PREV_JOB_ID:-}" ]; then
    log "PREV_JOB_ID not set; skipping cgroup lookup"
    return
  fi
  while read -r pid; do
    add_pid "$pid"
  done < <({ scontrol listpids "$PREV_JOB_ID" 2>/dev/null || true; } | awk 'NR>1 && $1 ~ /^[0-9]+$/ {print $1}')
}

collect_try80_cuda_pids() {
  while read -r pid; do
    [ -n "$pid" ] || continue
    local owner=""
    owner=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')
    [ "$owner" = "$TARGET_USER" ] || continue
    local cmdline=""
    cmdline=$(ps -o args= -p "$pid" 2>/dev/null || true)
    case "$cmdline" in
      *TFGEightiethTry80*|*train_try80.py*|*torchrun*)
        add_pid "$pid"
        ;;
    esac
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | awk 'NF')
}

collect_try80_process_pids() {
  while read -r pid; do
    add_pid "$pid"
  done < <(ps -u "$TARGET_USER" -o pid=,args= | awk '/TFGEightiethTry80|train_try80\.py|torchrun/ {print $1}')
}

kill_collected_pids() {
  local pids=("${!PIDS_TO_KILL[@]}")
  if [ "${#pids[@]}" -eq 0 ]; then
    log "no Try 80 residual pids found"
    return
  fi

  log "killing candidate pids: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 5
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

log "host=$(hostname) start=$(date --iso-8601=seconds) prev_job=${PREV_JOB_ID:-<unset>} user=$TARGET_USER"
log "nvidia-smi before"
nvidia-smi || true

collect_prev_job_pids
collect_try80_cuda_pids
collect_try80_process_pids
kill_collected_pids

log "user processes after kill"
ps -u "$TARGET_USER" -o pid,ppid,comm,args | egrep 'TFGEightiethTry80|train_try80|torchrun' || true

log "nvidia-smi after"
nvidia-smi || true
log "end=$(date --iso-8601=seconds)"
