# BrainCT-ICH: hemorrhage detection from whole head-CT series to a monitored, autoscaled cloud API

[![CI](https://github.com/NirmalKumar31/BrainCT-ICH-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/NirmalKumar31/BrainCT-ICH-MLOps/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch_(MPS·CUDA·DDP)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white">
  <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX_Runtime_INT8-005CED?style=for-the-badge&logo=onnx&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img alt="GitHub Actions" src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">
  <br>
  <img alt="GKE" src="https://img.shields.io/badge/GKE_Autopilot-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white">
  <img alt="Terraform" src="https://img.shields.io/badge/Terraform-844FBA?style=for-the-badge&logo=terraform&logoColor=white">
  <img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes_·_Helm-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white">
  <img alt="Prometheus" src="https://img.shields.io/badge/Prometheus_·_Alertmanager-E6522C?style=for-the-badge&logo=prometheus&logoColor=white">
  <img alt="Grafana" src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white">
  <img alt="Modal" src="https://img.shields.io/badge/Modal_(cloud_GPU)-111111?style=for-the-badge">
</p>

> **Research use only. Not a medical device. Not for diagnosis or any clinical decision.**

An end-to-end ML system that reads a whole non-contrast head CT series and reports, per slice
and per study, the probability of intracranial hemorrhage and its five subtypes:

- **data** streamed straight from a 144 GB archive into an HU cache (parallel HTTP ranges);
- **training** on GPU (Apple MPS locally; CUDA + multi-GPU DDP; Modal cloud-GPU pipeline),
  tracked in MLflow: a slice CNN plus a whole-series BiGRU;
- **evaluation** with patient-level bootstrap CIs, a 2×2 ablation, calibration and operating
  points, all fit on validation;
- **INT8 ONNX** certified through the serving code, on two CPU architectures;
- a **PyTorch-free FastAPI** service with CAM heatmaps, cross-request micro-batching, a startup
  numerics self-test, drift monitoring and live quality monitoring from ground-truth feedback;
- **deployment** with Terraform to GKE Autopilot (Helm, HPA, PDB, NetworkPolicy), Prometheus,
  Alertmanager and Grafana, load-tested in-cluster; plus kind + Helm end-to-end in CI.

It is a from-scratch rebuild of
[BrainCT-Hemorrhage-MLOps](https://github.com/pratox1112/BrainCT-Hemorrhage-MLOps). Every number
comes from [`docs/results.md`](docs/results.md); proof (screenshots, logs, raw outputs) is in
[`docs/evidence/`](docs/evidence/README.md); design choices and an incident write-up are in
[`docs/decisions.md`](docs/decisions.md).

## Highlights

| | Result (held-out test: 266 patients, 315 series, 10,934 slices) |
|---|---|
| **Deployed model** | macro ROC AUC **0.920** (95% CI 0.855–0.957, patient bootstrap); RSNA weighted log loss **0.123**; study-level AUC "any" **0.949** |
| **Series model vs single-slice CNN** | +0.018 macro AUC; subarachnoid 0.907 → **0.932**, subdural 0.916 → **0.939** |
| **Calibrated, with operating points** | ECE ≤ 0.012 per label (slice level); screening: 97.2% sensitivity at 68.3% specificity |
| **INT8** | same AUC as FP32 (0.920); 11.3 MB vs 44.7 MB; 32-slice series in **331 ms** vs 675 ms (FP32 ONNX) vs 1,027 ms (PyTorch) |
| **Served = certified, in the cloud** | 60 real series through GKE: mean \|Δp\| **0.0009** vs offline; study AUC 0.920 vs 0.922 |
| **Caught a cloud-only numerics bug** | INT8 diverged by 0.56 on GKE's AMD EPYC (no VNNI); fixed with uint8 weights + a per-pod startup self-test with FP32 fallback |
| **GKE autoscaling** | HPA 1 → 4 pods; 0 errors up to 16 concurrent clients; Terraform apply → test → destroy |
| **Monitoring that fired** | drift PSI and latency alerts fired and reached Alertmanager; live AUC 0.959 from 80 labelled studies via `/v1/feedback` |
| **CI** | lint, types, tests (94% coverage), Trivy, container smoke test, **kind + Helm e2e**, Terraform validate/tflint |

![Grafana on GKE during the load test](docs/evidence/gke/grafana_dashboard_under_load.png)

## Compared with the reference project

| | Reference (BrainCT-Hemorrhage-MLOps) | This repository |
|---|---|---|
| Model input | single slice | whole series (CNN + BiGRU series model); 2.5D tested |
| Training data | all 752,803 slices (A100, Slurm) | 9.5% sample on a laptop GPU; full-data Modal pipeline included |
| Test macro AUC | **0.977** (ResNet-50, full data) | 0.920 (ResNet-18, 9.5% data; different test set) |
| Probabilities | thresholds | Platt-calibrated, ECE reported, screening + study operating points |
| What is certified | exported model | the deployed chain, through the serving code, on ARM and x86 |
| Experiment tracking | Weights & Biases | MLflow |
| Multi-GPU | no | DDP (`torchrun`), verified with 2 processes |
| Serving | FastAPI, ONNX INT8, micro-batching | + series endpoint, CAM heatmaps, bundle checksums, numerics self-test with FP32 fallback |
| Drift monitoring | listed as missing | input + prediction PSI, alert |
| Quality monitoring | no | live AUC / Brier / ECE from delayed ground truth, alert |
| Alert routing | no | Alertmanager |
| Cloud | GKE Autopilot, Terraform, Helm, HPA | same, plus PDB, NetworkPolicy (Dataplane V2), least-privilege node SA |
| Deployment testing | manual | kind + Helm e2e in CI on every push; real-series e2e against certified outputs |

The reference model is more accurate: it trained a larger network on ten times the data. The
Modal pipeline here (`cloud/modal_pipeline.py`) runs the same code on all of RSNA with a
ResNet-50 on an A100; its results will be added when that run is done.

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data"]
        HF[(RSNA 16-bit PNG tar<br/>144 GB, Hugging Face)] -->|parallel HTTP ranges,<br/>tar resync| C[(HU cache<br/>int16, 224 px)]
        C --> S[patient split<br/>stratified by rarest subtype]
    end
    subgraph Train["Training: MPS · CUDA/DDP · Modal, tracked in MLflow"]
        S --> CNN[slice CNN<br/>HU windowing in-graph]
        CNN -->|embeddings + logits| SEQ[BiGRU series model]
    end
    subgraph Export["Export + certification"]
        SEQ --> Q[ONNX + static INT8<br/>uint8 weights]
        Q --> CERT[deployed chain on val/test:<br/>calibration, thresholds, CIs,<br/>drift reference, self-test golden]
    end
    subgraph Cloud["GKE Autopilot (Terraform) / kind (CI)"]
        CERT --> API[FastAPI pods: micro-batching,<br/>CAM, self-test, drift, feedback]
        API --> HPA[HPA · PDB · NetworkPolicy]
        API --> P[Prometheus] --> AM[Alertmanager]
        P --> G[Grafana]
    end
```

## The API

```bash
curl -F "files=@scan.zip" "http://127.0.0.1:8080/v1/predict/series?explain=true"
curl -X POST http://127.0.0.1:8080/v1/feedback -H 'content-type: application/json' \
     -d '{"request_id": "<x-request-id of the prediction>", "labels": {"any": 1, "subdural": 1}}'
```

| Endpoint | |
|---|---|
| `POST /v1/predict/series` | DICOM slices of one series (files or a zip) → per-slice probabilities, study summary (hemorrhage at the validation F1 threshold, screening flag, subtypes, key slice), warnings, optional CAM heatmap |
| `POST /v1/predict/slice` | one DICOM, CNN head only (separately calibrated) |
| `POST /v1/feedback`, `GET /v1/performance` | delayed ground truth → live study-level AUC / Brier / ECE |
| `GET /v1/model`, `GET /v1/drift` | model card (metrics, thresholds, precision, self-test), drift state |
| `GET /metrics`, `/health`, `/ready` | Prometheus metrics; liveness; readiness (503 until the bundle is verified and loaded) |

Example response, heatmap and OpenAPI screenshot: [`docs/evidence/`](docs/evidence/README.md).

## Repository layout

```
src/brainct/
  data/            streaming ingest (range requests, tar resync), patient split
  preprocessing/   DICOM -> HU (12-bit sign fix, slice geometry), pad + area resize
  training/        dataset, CNN + series model, train loops (MPS/CUDA, DDP), MLflow
  optimization/    ONNX export, INT8, certification, self-test golden, benchmark
  serving/         FastAPI app, ONNX Runtime inference, micro-batcher, series assembly
  monitoring/      drift (PSI vs training reference), live performance from feedback
  evaluation.py    calibration, thresholds, patient-bootstrap CIs, ablation tables
  metrics.py       AUC / AP / log loss / ECE / Platt in numpy
cloud/             Modal GPU pipeline, Cloud Build config, cloud requirements
infra/terraform/   GKE Autopilot, Artifact Registry, node IAM
infra/helm/        chart (HPA, PDB, NetworkPolicy, probes) + kind and GKE values
monitoring/        Prometheus, alert rules, Alertmanager, Grafana dashboard (Kustomize)
load_tests/        in-cluster load Job (kind, GKE) and Locust
scripts/           stand-in bundle, smoke / real-series e2e, quantization study, evidence, figures
tests/             unit, API and release tests; synthetic DICOMs only
docs/              results, decisions (incl. incident), evidence/, figures/, benchmark.json
```

## Run it

Python 3.12 and [uv](https://docs.astral.sh/uv/). Data and trained weights are not in the
repository (see below).

```bash
make setup lint test                     # environment, ruff + mypy, pytest
make data split train seq compare        # stream RSNA sample, split, CNN, series model, ablation
make export benchmark                    # ONNX + INT8 + certification -> artifacts/bundle
uv run pytest tests/regression           # release gates on the trained bundle
make image serve smoke                   # Docker image / local API / smoke test

# multi-GPU
uv run torchrun --nproc_per_node 4 -m brainct.training.train_cnn --config configs/cnn2d_resnet18.yaml

# full RSNA on a cloud GPU (Modal)
modal run cloud/modal_pipeline.py::ingest
modal run cloud/modal_pipeline.py::train --config full_resnet50_2d
modal run cloud/modal_pipeline.py::finish --config full_resnet50_2d

# GKE (costs money while it runs)
terraform -chdir=infra/terraform apply
gcloud builds submit --config cloud/cloudbuild.yaml --substitutions=_IMAGE=<registry>/brainct-api:0.2.1 <context>
helm install brainct infra/helm/brainct -f infra/helm/brainct/values-gke.yaml --set image.repository=<registry>/brainct-api
kubectl apply -k monitoring
uv run python scripts/e2e_real_series.py http://127.0.0.1:8080 --tol 0.05 --feedback
terraform -chdir=infra/terraform destroy
```

Without the data, `make standin image BUNDLE=artifacts/standin` builds the service with a
weights-free stand-in model of the same interface; that is what CI deploys.

## Known limitations

- **Epidural is not usable:** 39 test slices from 5 patients; AUC 0.760 (CI 0.37–0.95), AP 0.014.
- **Small sample of one source:** 9.5% of RSNA 2019, no external validation; ablation
  differences are smaller than their CIs (one seed each).
- **Series-model training embeddings are in-sample** (from the CNN trained on those slices).
- **Study-level probabilities are not re-calibrated:** slice-level ECE ≤ 0.012, but the max over
  a series showed ECE 0.08–0.12 on 80 live-labelled studies.
- **Feedback is joined per pod:** predictions awaiting ground truth live in each pod's memory,
  so with several replicas feedback must reach the pod that served the request. A shared store
  (e.g. Redis) or an offline join from prediction logs is needed in production.
- **CAM heatmaps come from the CNN**, not the series model; they show attention, not a segmentation.
- **CPU-only serving.** On GKE, a 2-vCPU EPYC pod serves ~1.4 series/s; the 4th Autopilot
  pod took ~4–5 minutes to come up (capacity provisioning).

## Data and license

- Data: [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
  (2019), via its [16-bit PNG release](https://huggingface.co/datasets/ianpan/rsna-intracranial-hemorrhage-16bit-png).
  Its terms apply. **No patient images, caches or trained weights are distributed here**; the
  trained image lived only in a private registry that was deleted with the cluster.
- Code: [MIT](LICENSE).
- Reference project and the original 2025 notebooks: Pratosh Karthikeyan and Nirmal,
  [BrainCT-Hemorrhage-MLOps](https://github.com/pratox1112/BrainCT-Hemorrhage-MLOps).
