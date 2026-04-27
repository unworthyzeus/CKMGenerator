#!/usr/bin/env python3
"""Upload and submit a single Try 80 1-GPU job."""
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
TRAIN_SLURM = "cluster/run_eightieth_try80_1gpu.slurm"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/try80_joint_big.yaml")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--node", default=TARGET_NODE)
    parser.add_argument("--base-master-port", type=int, default=32780)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--wipe-outputs", action="store_true")
    parser.add_argument("--depend-on-job", default="")
    parser.add_argument("--dependency-type", default="afterany", choices=("afterany", "afterok"))
    parser.add_argument("--slurm-file", default=TRAIN_SLURM)
    parser.add_argument("--job-name", default="t80-1gpu")
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
        "hostname": args.host,
        "username": args.user,
        "timeout": 30,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if password:
        connect_kwargs["password"] = password
    if args.ssh_key:
        connect_kwargs["key_filename"] = args.ssh_key
    client.connect(**connect_kwargs)

    try:
        dep = f"--dependency={args.dependency_type}:{args.depend_on_job} " if args.depend_on_job else ""
        exports = [
            f"CONFIG_PATH={args.config}",
            "TRAIN_SCRIPT=train_try80.py",
            "MASTER_ADDR=127.0.0.1",
            f"MASTER_PORT={args.base_master_port}",
            f"RDZV_ID={args.job_name}",
        ]
        if args.wipe_outputs:
            exports.append("WIPE_OUTPUTS=1")
        if args.no_resume:
            exports.append("RESUME_CHECKPOINT=")
        command = (
            f"cd {REMOTE_DIR} && "
            f"sbatch --nodelist={args.node} {dep}"
            f"-J {args.job_name} --export=ALL,{','.join(exports)} {args.slurm_file}"
        )
        out, err = remote_exec(client, command)
        if err:
            raise RuntimeError(err)
        print(f"Submitted Try 80 1-GPU job {parse_job_id(out)}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
