"""Cluster dispatcher — runs on the personal laptop.

SSHes to kudu-taught via warwick-cluster alias (ControlMaster must be active).
On SSH failure: falls back to llama3.1:8b locally.
"""
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from backend.config import settings

LOCAL_TEMP = Path(settings.local_temp_dir)
CLUSTER_HOST = settings.cluster_ssh_alias          # warwick-cluster
CLUSTER_USER = settings.cluster_user
CLUSTER_PROJECT = settings.cluster_project_dir
CLUSTER_INPUT_DIR = settings.cluster_shared_input
CLUSTER_OUTPUT_DIR = settings.cluster_shared_output


class ClusterUnavailableError(Exception):
    pass


def _build_full_prompt(payload: dict) -> str:
    """Reconstruct the same full-prompt format the cluster-side inference
    script (cluster_inference.py) and moe_generator.py's local-generation
    branch both use, so a cluster timeout/failure falls back to a fully
    grounded local answer (retrieved context, constraints, grounding rules)
    instead of the bare user query.
    """
    system_prompt = payload.get("system_prompt", "")
    query = payload.get("query", "")
    return f"{system_prompt}\n\nUser: {query}\n\nAssistant:" if system_prompt else query


def sync_project_to_cluster() -> None:
    """rsync project source (no data/, no .env) to kudu-taught."""
    project_root = Path(__file__).parent.parent.parent
    cmd = [
        "rsync", "-az",
        "--exclude=.env",
        "--exclude=data/",
        "--exclude=__pycache__",
        "--exclude=.git",
        "--exclude=.venv",
        f"{project_root}/",
        f"{CLUSTER_HOST}:{CLUSTER_PROJECT}/",
    ]
    subprocess.run(cmd, check=True, timeout=120)


def push_input(payload: dict) -> str:
    """Write payload JSON, scp it to the cluster input dir. Returns job_uuid."""
    job_uuid = str(uuid.uuid4())
    LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_TEMP / f"input_{job_uuid}.json"
    local_path.write_text(json.dumps(payload), encoding="utf-8")

    remote_path = f"{CLUSTER_INPUT_DIR}/input_{job_uuid}.json"
    result = subprocess.run(
        ["scp", str(local_path), f"{CLUSTER_HOST}:{remote_path}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise ClusterUnavailableError(f"scp failed: {result.stderr}")
    return job_uuid


def submit_slurm_job(sbatch_file: str, env_vars: dict[str, str]) -> str:
    """Submit sbatch job on kudu-taught. Returns Slurm job ID."""
    export_str = ",".join(f"{k}={v}" for k, v in env_vars.items())
    result = subprocess.run(
        [
            "ssh", CLUSTER_HOST,
            f"sbatch --export={export_str} {sbatch_file}",
        ],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise ClusterUnavailableError(f"sbatch failed: {result.stderr}")
    # Output: "Submitted batch job 12345"
    return result.stdout.strip().split()[-1]


def poll_job(job_uuid: str, timeout_seconds: int | None = None) -> bool:
    """Poll for the output file's existence on the cluster (1s intervals).

    Returns True as soon as the paired output JSON appears within
    timeout_seconds (default: settings.cluster_timeout_seconds, 600s).

    Checks the output path directly via the same SSH remote-access
    mechanism pull_output() uses to fetch it, rather than polling Slurm
    queue state (squeue): a job can leave the queue slightly before its
    output file is fully flushed and visible on the shared filesystem, so
    file-existence is the actual condition the caller needs satisfied
    before pull_output() can succeed.
    """
    timeout_seconds = timeout_seconds or settings.cluster_timeout_seconds
    remote_path = f"{CLUSTER_OUTPUT_DIR}/output_{job_uuid}.json"
    start = time.time()
    while time.time() - start < timeout_seconds:
        result = subprocess.run(
            ["ssh", CLUSTER_HOST, f"test -f {remote_path} && echo EXISTS"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip() == "EXISTS":
            return True
        time.sleep(1)
    return False


def pull_output(job_uuid: str) -> dict[str, Any]:
    """scp output JSON from cluster back to laptop."""
    remote_path = f"{CLUSTER_OUTPUT_DIR}/output_{job_uuid}.json"
    local_path = LOCAL_TEMP / f"output_{job_uuid}.json"
    result = subprocess.run(
        ["scp", f"{CLUSTER_HOST}:{remote_path}", str(local_path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise ClusterUnavailableError(f"scp pull failed: {result.stderr}")
    return json.loads(local_path.read_text(encoding="utf-8"))


def dispatch_to_cluster_sync(
    payload: dict,
    fallback_fn,
    timeout_seconds: int = 600,
    sbatch_file: str = "mixtral_inference.sbatch",
) -> str:
    """
    Synchronous dispatch — safe to call from asyncio.to_thread().
    Uses a longer default timeout than the fire-and-forget path.
    """
    try:
        job_uuid = push_input(payload)
        job_id = submit_slurm_job(
            f"{CLUSTER_PROJECT}/jobs/{sbatch_file}",
            {
                "INFER_INPUT": f"{CLUSTER_INPUT_DIR}/input_{job_uuid}.json",
                "INFER_OUTPUT": f"{CLUSTER_OUTPUT_DIR}/output_{job_uuid}.json",
            },
        )
        done = poll_job(job_uuid, timeout_seconds=timeout_seconds)
        if not done:
            raise TimeoutError(f"Cluster job {job_id} timed out after {timeout_seconds}s")
        result = pull_output(job_uuid)
        return result["response"]
    except Exception as exc:
        # Cluster/SSH access is inherently flaky (ControlMaster drop, node
        # preemption, sbatch queue full, etc.) — fail open to local generation
        # via fallback_fn rather than surfacing an error to the caller. Pass
        # the full assembled prompt (system prompt + query), not the bare
        # query, so the degraded answer is still grounded in retrieved
        # context/constraints/grounding rules.
        print(f"[cluster_dispatcher] Falling back to local: {exc}")
        return fallback_fn(_build_full_prompt(payload))


async def dispatch_to_cluster(payload: dict, fallback_fn, sbatch_file: str = "mixtral_inference.sbatch") -> str:
    """
    Full dispatch cycle. On any error, call fallback_fn(payload) and return its result.

    fallback_fn: callable(payload) -> str  (e.g. llm_medium.invoke)
    """
    try:
        job_uuid = push_input(payload)
        job_id = submit_slurm_job(
            f"{CLUSTER_PROJECT}/jobs/{sbatch_file}",
            {
                "INFER_INPUT": f"{CLUSTER_INPUT_DIR}/input_{job_uuid}.json",
                "INFER_OUTPUT": f"{CLUSTER_OUTPUT_DIR}/output_{job_uuid}.json",
            },
        )
        done = poll_job(job_uuid, timeout_seconds=settings.cluster_timeout_seconds)
        if not done:
            raise TimeoutError(f"Cluster job {job_id} timed out")
        result = pull_output(job_uuid)
        return result["response"]

    except Exception as exc:
        # Same resilience contract as dispatch_to_cluster_sync above: any failure
        # in the SSH/Slurm round-trip degrades to local generation instead of
        # failing the request outright. Pass the full assembled prompt (system
        # prompt + query), not the bare query, so the fallback answer is still
        # grounded in retrieved context/constraints/grounding rules.
        print(f"[cluster_dispatcher] Falling back to local: {exc}")
        # fallback_fn may be sync — call directly
        return fallback_fn(_build_full_prompt(payload))
