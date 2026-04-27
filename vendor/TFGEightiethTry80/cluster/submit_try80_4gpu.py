#!/usr/bin/env python3
"""Submit chained 4-GPU Slurm jobs for Try 80 (train -> cleanup -> ... 5 times)."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import paramiko
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "paramiko"], check=True)
    import paramiko

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "TFGEightiethTry80"
REMOTE_DIR = "/scratch/nas/3/gmoreno/TFGpractice/TFGEightiethTry80"
HOST = "sert.ac.upc.edu"
USER = "gmoreno"
TARGET_NODE = "sert-2001"

TRAIN_SLURM = "cluster/run_eightieth_try80_4gpu.slurm"
CLEANUP_SLURM = "cluster/run_eightieth_try80_cleanup_sert2001_1gpu.slurm"
DEFAULT_BASE_MASTER_PORT = 32780


def run_local(command: list[str]) -> None:
    print("LOCAL>", " ".join(command))
    subprocess.run(command, check=True, cwd=str(ROOT))


def parse_job_id(text: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", text)
    if not match:
        raise RuntimeError(f"Could not parse job id from: {text!r}")
    return match.group(1)


def remote_exec(client: paramiko.SSHClient, command: str, *, check: bool = True) -> tuple[str, str]:
    print("REMOTE>", command)
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err:
        print(err)
    if check and err:
        raise RuntimeError(err)
    return out, err


def remote_sbatch(client: paramiko.SSHClient, command: str) -> str:
    out, err = remote_exec(client, f"cd {REMOTE_DIR} && {command}")
    if err:
        raise RuntimeError(err)
    return parse_job_id(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit 5 chained Try 80 4-GPU jobs (with cleanup between)."
    )
    parser.add_argument("--config", default="experiments/try80_joint_big.yaml")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--cancel-all", action="store_true")
    parser.add_argument("--cancel-job-ids", default="")
    parser.add_argument("--base-master-port", type=int, default=DEFAULT_BASE_MASTER_PORT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--wipe-outputs", action="store_true")
    parser.add_argument("--depend-on-job", default="",
                        help="Optional existing Slurm job id to depend on before starting the chain.")
    parser.add_argument("--dependency-type", default="afterany", choices=("afterany", "afterok"),
                        help="Dependency type used with --depend-on-job (default: afterany).")
    parser.add_argument("--chain-length", type=int, default=5,
                        help="Number of times to run the job in sequence.")
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    if not args.skip_upload:
        run_local([
            sys.executable,
            str(ROOT / "cluster" / "upload_and_submit_experiments.py"),
            "--local-dir", str(LOCAL_DIR),
            "--upload-only",
            "--skip-datasets",
        ])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict[str, Any] = {
        "hostname": args.host, "username": args.user, "timeout": 30,
        "allow_agent": True, "look_for_keys": True,
    }
    if password:
        connect_kwargs["password"] = password
    if args.ssh_key:
        connect_kwargs["key_filename"] = args.ssh_key
    client.connect(**connect_kwargs)

    try:
        if args.cancel_all:
            remote_exec(client, f"scancel -u {args.user}", check=False)
        for jid in (s.strip() for s in str(args.cancel_job_ids).split(",") if s.strip()):
            remote_exec(client, f"scancel {jid}", check=False)
        remote_exec(client, f"squeue -u {args.user}", check=False)

        sbatch_prefix = f"sbatch --nodelist={args.node}"
        dep_job = str(args.depend_on_job).strip()
        current_dep = f"--dependency={args.dependency_type}:{dep_job} " if dep_job else ""
        submitted: list[tuple[str, str]] = []

        for i in range(args.chain_length):
            job_name = f"t80-4gpu-chain{i+1}"
            exports = [
                f"CONFIG_PATH={args.config}",
                "TRAIN_SCRIPT=train_try80.py",
                "MASTER_ADDR=127.0.0.1",
                f"MASTER_PORT={args.base_master_port + i * 4}",
                f"RDZV_ID={job_name}",
            ]
            
            # Usually only the first job in a chain should wipe outputs if requested
            # otherwise subsequent jobs will wipe the first job's outputs!
            if args.wipe_outputs and i == 0:
                exports.append("WIPE_OUTPUTS=1")
            
            if args.no_resume and i == 0:
                exports.append("RESUME_CHECKPOINT=")
            
            train_cmd = (
                f"{sbatch_prefix} {current_dep}-J {job_name} "
                f"--export=ALL,{','.join(exports)} "
                f"{TRAIN_SLURM}"
            )
            train_job_id = remote_sbatch(client, train_cmd)
            submitted.append((job_name, train_job_id))

            cleanup_name = f"t80-cleanup{i+1}"
            cleanup_cmd = (
                f"{sbatch_prefix} --dependency=afterany:{train_job_id} "
                f"-J {cleanup_name} "
                f"--export=ALL,PREV_JOB_ID={train_job_id} "
                f"{CLEANUP_SLURM}"
            )
            cleanup_job_id = remote_sbatch(client, cleanup_cmd)
            submitted.append((cleanup_name, cleanup_job_id))
            current_dep = f"--dependency=afterany:{cleanup_job_id} "

        print("\n=== Submitted jobs (train -> cleanup -> train -> ...) ===")
        for name, jid in submitted:
            tag = "  [cleanup]" if "cleanup" in name else "  [train]  "
            print(f"{tag} {name}: job {jid}")
        print(f"\nTotal: {len(submitted)} jobs ({args.chain_length} trains + {args.chain_length} cleanups).")
    finally:
        client.close()


if __name__ == "__main__":
    main()
