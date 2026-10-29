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

RUNS, LOCAL_ROOT = Path("/runs"), Path("/local")
# SQLite must not live on a network volume (locking): work on a local copy, sync it back.
MLFLOW_LOCAL = Path("/local/mlflow.db")
MLFLOW = f"sqlite:///{MLFLOW_LOCAL}"


def mlflow_pull() -> None:
    MLFLOW_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if (RUNS / "mlflow.db").exists():
        shutil.copyfile(RUNS / "mlflow.db", MLFLOW_LOCAL)


def mlflow_push() -> None:
    if MLFLOW_LOCAL.exists():
        shutil.copyfile(MLFLOW_LOCAL, RUNS / "mlflow.db")


def sh(*args: str, env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd="/app", env={**os.environ, **(env or {})})


@app.function(cpu=16, memory=32768, timeout=3 * 3600, volumes={"/data": data_vol})
def ingest(dataset: str = "rsna", regions: int = 32, gib_per_region: float = 4.3) -> None:
    """Whole tar (32 x 4.3 GiB > 144 GB): every series kept by exactly one region."""
    data = Path("/data") / dataset
    sh(
        sys.executable,
        "-m",
        "brainct.data.ingest",
        "--out",
        str(data),
        "--regions",
        str(regions),
        "--gib-per-region",
        str(gib_per_region),
    )
    sh(sys.executable, "-m", "brainct.data.split", "--data", str(data))
    data_vol.commit()


def stage_local(dataset: str) -> Path:
    """Copy the cache from the volume to local NVMe (once per container), 32 files at a time."""
    from concurrent.futures import ThreadPoolExecutor

    src, dst = Path("/data") / dataset, LOCAL_ROOT / dataset
    if not (dst / "slices.csv").exists():
        t0 = time.time()
        (dst / "series").mkdir(parents=True, exist_ok=True)
        files = list((src / "series").glob("*.npy"))
        with ThreadPoolExecutor(32) as pool:
            list(pool.map(lambda f: shutil.copyfile(f, dst / "series" / f.name), files))
        for name in ("slices.csv", "splits.csv", "manifest.json"):
            shutil.copyfile(src / name, dst / name)
        print(f"staged {len(files)} series to local disk in {time.time() - t0:.0f} s", flush=True)
    return dst


@app.function(
    gpu=GPU,
    cpu=16,
    memory=65536,
    ephemeral_disk=512 * 1024,
    timeout=10 * 3600,
    retries=2,
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def train(config: str = "full_resnet50_2d", dataset: str = "rsna", overrides: str = "") -> None:
    local = stage_local(dataset)
    sh("nvidia-smi")
    sets = [f"data_dir={local}", f"out_dir={RUNS}", *overrides.split()]

    # Commit the runs volume every few minutes so checkpoints survive preemption.
    import threading

    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(300):
            mlflow_push()
            runs_vol.commit()

    mlflow_pull()
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
        mlflow_push()
        runs_vol.commit()


@app.function(
    cpu=32,
    memory=65536,
    ephemeral_disk=512 * 1024,
    timeout=6 * 3600,
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def finish(
    config: str = "full_resnet50_2d",
    dataset: str = "rsna",
    export: bool = True,
    overrides: str = "",
) -> None:
    """Series model and CNN-vs-series evaluation; with export, ONNX/INT8 + certification."""
    local = stage_local(dataset)
    sets = [f"data_dir={local}", f"out_dir={RUNS}", *overrides.split()]
    from brainct.training.config import TrainConfig

    run = TrainConfig.load(f"/app/configs/{config}.yaml", sets).run_dir()
    env = {"MLFLOW_TRACKING_URI": MLFLOW}
    cfg = f"/app/configs/{config}.yaml"
    mlflow_pull()
    sh(sys.executable, "-m", "brainct.training.train_seq", "--config", cfg, "--set", *sets, env=env)
    mlflow_push()
    runs_vol.commit()
    sh(sys.executable, "-m", "brainct.evaluation", "--compare", f"{run}:cnn", f"{run}:seq")
    runs_vol.commit()
    if not export:
        return
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
