# Results

Every number here was measured in this repository (2026-10-29) on an Apple M4 laptop
(10-core CPU, 16 GB), GitHub Actions, and GKE Autopilot (us-central1). How each number was
produced is stated next to it; raw proof is in [`evidence/`](evidence/README.md).

## Data

| | |
|---|---|
| Source | RSNA 2019 ICH, 16-bit PNG release (`ianpan/rsna-intracranial-hemorrhage-16bit-png@c40c72b`) |
| Sample | 2,065 complete series / 71,736 slices / 1,770 patients (9.5% of RSNA), streamed from 16 tar regions in 141 s |
| Split | patient-level 70/15/15, seed 2026, stratified by rarest subtype |
| Test set | 266 patients, 315 series, 10,934 slices; 1,853 hemorrhage slices, 134 hemorrhage series |
| Test positives (slices) | epidural 39 (5 patients), intraparenchymal 678, intraventricular 460, subarachnoid 621, subdural 904 |

## Ablation: context from 2.5D input vs a series model

Same backbone (ResNet-18, ImageNet init), schedule (5 epochs, balanced sampling), seed and split;
checkpoint chosen on validation; Platt calibration fit on validation; test scored once. 95% CIs
resample patients (1,000×). Training curves: [`evidence/mlflow_training_curves.png`](evidence/mlflow_training_curves.png).

| Model | Macro AUC (95% CI) | Weighted log loss (95% CI) | Study AUC, any (95% CI) | SAH AUC | SDH AUC |
|---|---|---|---|---|---|
| Single-slice CNN (reference project's input) | 0.902 (0.844–0.940) | 0.133 (0.108–0.165) | 0.953 (0.926–0.975) | 0.907 | 0.916 |
| **Single-slice CNN + series model (deployed)** | **0.920 (0.858–0.956)** | **0.122 (0.097–0.152)** | 0.953 (0.925–0.975) | **0.932** | **0.939** |
| 2.5D CNN | 0.897 (0.841–0.934) | 0.142 (0.114–0.176) | 0.944 (0.915–0.967) | 0.892 | 0.897 |
| 2.5D CNN + series model | 0.915 (0.856–0.951) | 0.128 (0.102–0.159) | 0.946 (0.919–0.968) | 0.917 | 0.936 |

The series model adds ~+0.018 macro AUC to both CNNs, most on subarachnoid (+0.025) and subdural
(+0.023), the reference project's weakest subtypes. 2.5D input did not help at this data scale.
Differences are smaller than the CIs (one seed each).

## Deployed chain (INT8 CNN → series model), certified through the serving code

`make export`: ONNX (parity vs PyTorch 1.6e-5), static INT8 with uint8 weights, then the
deployed chain runs through `serving.model.InferenceModel` on validation (fit) and test (score).

| Precision | Macro AUC (95% CI) | Weighted log loss | Study AUC (any) | CNN file |
|---|---|---|---|---|
| FP32 ONNX | 0.920 | 0.122 | 0.953 | 44.7 MB |
| **INT8 ONNX, uint8 weights (deployed)** | **0.920 (0.855–0.957)** | **0.123 (0.099–0.153)** | **0.949 (0.919–0.971)** | **11.3 MB** |
| INT8, int8 weights + reduce_range (rejected) | 0.843 | 2.213 | 0.946 | 11.3 MB |

Per label, deployed chain, test (slice level, thresholds F1-optimal on validation):

| Label | Positive slices | AUC (95% CI) | AP | ECE | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| any | 1,853 | 0.951 (0.936–0.964) | 0.840 | 0.012 | 0.792 | 0.936 |
| epidural | 39 | 0.760 (0.372–0.948) | 0.014 | 0.003 | 0.026 | 0.981 |
| intraparenchymal | 678 | 0.967 (0.954–0.978) | 0.781 | 0.005 | 0.652 | 0.986 |
| intraventricular | 460 | 0.974 (0.962–0.986) | 0.784 | 0.006 | 0.691 | 0.987 |
| subarachnoid | 621 | 0.931 (0.903–0.957) | 0.514 | 0.011 | 0.498 | 0.970 |
| subdural | 904 | 0.939 (0.917–0.960) | 0.614 | 0.010 | 0.629 | 0.952 |

| Operating point for "any" (chosen on validation) | Level | Sensitivity | Specificity | PPV |
|---|---|---|---|---|
| Screening (≥95% sensitivity on validation) | slice | 0.972 | 0.683 | 0.385 |
| F1-optimal | study (max over slices) | 0.940 | 0.790 | 0.768 |

![ROC](figures/roc_test.png) ![Calibration](figures/calibration_test.png)

## Latency (CPU, 4 threads, Apple M4)

`make benchmark` → [`benchmark.json`](benchmark.json). 32-slice series; the chain runs the CNN
in batches of 16, then the series model.

| Engine | 1 slice p50 / p95 | 32-slice series p50 / p95 |
|---|---|---|
| PyTorch FP32 | 11.7 / 12.2 ms | 1,027 / 1,111 ms (CNN only) |
| ONNX Runtime FP32 | 12.8 / 13.0 ms | 675 / 712 ms |
| **ONNX Runtime INT8 (uint8 weights)** | **8.2 / 8.6 ms** | **331 / 399 ms** |

INT8 is 2.0× faster than FP32 ONNX and 3.1× faster than PyTorch per series. The first INT8
build (int8 weights) was faster on ARM (257 ms) but numerically wrong on x86 without VNNI (see
the incident below); uint8 weights trade ~30% ARM speed for identical behaviour on every CPU.

## Kubernetes, local (kind on a 4-vCPU Colima VM)

- **Serving = evaluation:** 60 real held-out series uploaded as shuffled DICOMs through the
  Service matched the offline certified probabilities within **5.0e-6** (same CPU family).
- **Autoscaling** (in-cluster Job, 32-slice series): HPA 1 → 3 pods in 31 s; 2.17 → 3.12
  series/s; 0 errors up to 8 clients. Ceiling = the VM's 4 vCPUs
  ([`evidence/kind_load_steps.md`](evidence/kind_load_steps.md)).

## Kubernetes, cloud (GKE Autopilot via Terraform, AMD EPYC 7B13 nodes)

Provisioned with `infra/terraform` (13 resources: Autopilot cluster with Dataplane V2, Artifact
Registry with cleanup policy, least-privilege node service account; cluster ready in 6 min 52 s),
image built on amd64 by Cloud Build, deployed with the same Helm chart, then destroyed.

**Correctness on the cloud CPU.** 60 real held-out series through the GKE Service: mean
|Δp| vs certified **0.0009**, max 0.027; study AUC 0.920 (offline, same series: 0.922). The first
deployment did *not* pass this check (max |Δp| 0.56): see the incident below.

**Autoscaling under in-cluster load** (Job, 32-slice synthetic series, 90 s steps, last 60 s measured):

| Concurrent clients | Series/s | p50 ms | p95 ms | Errors | Ready pods |
|---|---|---|---|---|---|
| 1 | 1.42 | 701 | 813 | 0 | 1 → 2 |
| 2 | 2.00 | 797 | 2,002 | 0 | 2 → 3 |
| 4 | 2.52 | 1,273 | 3,566 | 0 | 3 |
| 8 | 2.57 | 1,406 | 8,528 | 0 | 3 (HPA wants 4) |
| 12 | 3.08 | 2,086 | 10,609 | 0 | 3 → 4 |
| 16 | 2.65 | 1,487 | 16,448 | 0 | 4 |

HPA asked for a 2nd and 3rd pod within 50 s (ready by 150 s); the 4th pod took ~4–5 minutes
because Autopilot had to add capacity. One 2-vCPU EPYC pod serves ~1.4 series/s (slower than
the M4). Past ~3 series/s the extra clients only queue: p95 grows, nothing fails. Timeline:
[`evidence/gke/hpa_timeline.txt`](evidence/gke/hpa_timeline.txt).

**Monitoring and alerting, live.** Grafana under load:
[`evidence/gke/grafana_dashboard_under_load.png`](evidence/gke/grafana_dashboard_under_load.png).
The synthetic load images are not real CT; drift PSI rose to 3–12 on every feature and
`BrainCTInputDrift` fired; p95 above 5 s at 16 clients fired `BrainCTHighLatency`; both reached
Alertmanager (6 alert instances). `BrainCTLiveQualityDrop` stayed inactive.

**Live model quality from delayed ground truth.** For 80 real series the true study labels were
posted to `/v1/feedback`; the service's live study AUC for "any" was **0.959** (40 positive),
equal to the offline AUC of the same series
([`evidence/gke/live_performance_feedback.json`](evidence/gke/live_performance_feedback.json)).
Study-level ECE was 0.08–0.12: calibration is fit per slice, and the max over a series is not
re-calibrated (see limitations).

## Multi-GPU training path

`torchrun --nproc_per_node 2` with the gloo backend on CPU: both ranks trained their shard
(8 steps each), rank 0 validated, checkpointed and wrote predictions
([`evidence/ddp_gloo_2proc.log`](evidence/ddp_gloo_2proc.log)). On CUDA the same code uses NCCL.

## CI (GitHub Actions, every push)

Lint, format, mypy, the service-side tests with an 85% coverage gate (94% measured; 89 tests
locally including PyTorch and trained-bundle release tests), lockfile/requirements drift check;
image build, Trivy scan, no-PyTorch and non-root assertions, read-only-root smoke test; kind +
Helm end-to-end with a rolling restart and monitoring manifests; Terraform fmt/validate/tflint.
