"""The full-data pipeline on Modal: CPU ingest, GPU training, CPU export. Same code as local.

    modal run cloud/modal_pipeline.py::ingest                 # all of RSNA -> Volume (~CPU only)
    modal run cloud/modal_pipeline.py::train --config full_resnet50_2d     # A100
    modal run cloud/modal_pipeline.py::finish --config full_resnet50_2d    # series model,
                                                                           # export, reports
    modal volume get brainct-runs full_resnet50_2d/bundle artifacts/ ...   # download results

Storage: `brainct-data` (HU cache, ~75 GB) and `brainct-runs` (checkpoints, predictions,
MLflow, bundles). Training copies the cache to the container's local NVMe first: random
3-slice reads from a network volume would starve the GPU. Training resumes from last.pt, so a
preempted container (retries=2) continues where it stopped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("BRAINCT_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(ROOT / "cloud/requirements-train.txt"))
    .env({"MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("brainct")
    .add_local_dir(ROOT / "configs", "/app/configs")
)
data_vol = modal.Volume.from_name("brainct-data", create_if_missing=True, version=2)
runs_vol = modal.Volume.from_name("brainct-runs", create_if_missing=True, version=2)
app = modal.App("brainct", image=image)

DATA, RUNS, LOCAL = Path("/data/rsna"), Path("/runs"), Path("/local/rsna")
MLFLOW = f"sqlite:///{RUNS}/mlflow.db"


def sh(*args: str, env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd="/app", env={**os.environ, **(env or {})})


@app.function(cpu=16, memory=32768, timeout=3 * 3600, volumes={"/data": data_vol})
def ingest(regions: int = 32, gib_per_region: float = 4.3) -> None:
    """Whole tar (32 x 4.3 GiB > 144 GB): every series kept by exactly one region."""
    sh(
        sys.executable,
        "-m",
        "brainct.data.ingest",
        "--out",
        str(DATA),
        "--regions",
        str(regions),
        "--gib-per-region",
        str(gib_per_region),
    )
    sh(sys.executable, "-m", "brainct.data.split", "--data", str(DATA))
    data_vol.commit()


def stage_local() -> Path:
    """Copy the cache to local disk (once per container)."""
    if not (LOCAL / "slices.csv").exists():
        t0 = time.time()
        LOCAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(DATA, LOCAL, dirs_exist_ok=True)
        print(f"staged cache to local disk in {time.time() - t0:.0f} s", flush=True)
    return LOCAL


@app.function(
    gpu=GPU,
    cpu=16,
    memory=65536,
    ephemeral_disk=200 * 1024,
    timeout=10 * 3600,
    retries=2,
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def train(config: str = "full_resnet50_2d", overrides: str = "") -> None:
    local = stage_local()
    sh("nvidia-smi")
    sets = [f"data_dir={local}", f"out_dir={RUNS}", *overrides.split()]

    # Commit the runs volume every few minutes so checkpoints survive preemption.
    import threading

    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(300):
            runs_vol.commit()

    threading.Thread(target=committer, daemon=True).start()
    try:
        sh(
            sys.executable,
            "-m",
            "brainct.training.train_cnn",
            "--config",
            f"/app/configs/{config}.yaml",
            "--set",
            *sets,
            env={"MLFLOW_TRACKING_URI": MLFLOW},
        )
    finally:
        stop.set()
        runs_vol.commit()


@app.function(
    cpu=32,
    memory=131072,
    ephemeral_disk=200 * 1024,
    timeout=6 * 3600,
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def finish(config: str = "full_resnet50_2d") -> None:
    """Series model, 2-model evaluation, ONNX/INT8 export + certification, benchmark."""
    local = stage_local()
    run = RUNS / config
    env = {"MLFLOW_TRACKING_URI": MLFLOW}
    sets = [f"data_dir={local}", f"out_dir={RUNS}"]
    cfg = f"/app/configs/{config}.yaml"
    sh(sys.executable, "-m", "brainct.training.train_seq", "--config", cfg, "--set", *sets, env=env)
    runs_vol.commit()
    sh(sys.executable, "-m", "brainct.evaluation", "--compare", f"{run}:cnn", f"{run}:seq")
    sh(
        sys.executable,
        "-m",
        "brainct.optimization.export",
        "--run",
        str(run),
        "--out",
        str(run / "bundle"),
        "--data-dir",
        str(local),
    )
    sh(
        sys.executable,
        "-m",
        "brainct.optimization.benchmark",
        "--run",
        str(run),
        "--work",
        str(run / "bundle.work"),
        "--out",
        str(run / "benchmark.json"),
    )
    # Keep what is needed for figures; drop the large FP32/intermediate files.
    for f in (run / "bundle.work").glob("cnn.*.onnx"):
        f.unlink()
    runs_vol.commit()


@app.local_entrypoint()
def main() -> None:
    print(__doc__)
