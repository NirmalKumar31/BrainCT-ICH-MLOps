# Results

Every number here was measured in this repository on 2026-10-29, on an Apple M4 laptop
(10-core CPU, 16 GB) plus GitHub Actions. How each was produced is stated next to it.

## Data

| | |
|---|---|
| Source | RSNA 2019 ICH, 16-bit PNG release (`ianpan/rsna-intracranial-hemorrhage-16bit-png@c40c72b`) |
| Sample | 2,065 complete series / 71,736 slices / 1,770 patients, streamed from 16 evenly spaced tar regions in 141 s (`make data`) |
| Split | patient-level 70/15/15, seed 2026, stratified by rarest subtype (`make split`) |
| Test set | 266 patients, 315 series, 10,934 slices; 1,853 hemorrhage slices, 134 hemorrhage series |
| Test positives by subtype (slices) | epidural 39 (5 patients), intraparenchymal 678, intraventricular 460, subarachnoid 621, subdural 904 |

## Ablation: context from 2.5D input vs from a series model

Same backbone (ResNet-18, ImageNet init), schedule (5 epochs, balanced sampling), seed and
split; checkpoint chosen on validation; calibration fit on validation; test scored once.
PyTorch predictions (`make compare`). 95% CIs resample patients (1,000×).

| Model | Macro AUC (95% CI) | Weighted log loss (95% CI) | Study AUC, any (95% CI) | Subarachnoid AUC | Subdural AUC |
|---|---|---|---|---|---|
| Single-slice CNN (reference project's input) | 0.902 (0.844–0.940) | 0.133 (0.108–0.165) | 0.953 (0.926–0.975) | 0.907 | 0.916 |
| **Single-slice CNN + series model (deployed)** | **0.920 (0.858–0.956)** | **0.122 (0.097–0.152)** | 0.953 (0.925–0.975) | **0.932** | **0.939** |
| 2.5D CNN | 0.897 (0.841–0.934) | 0.142 (0.114–0.176) | 0.944 (0.915–0.967) | 0.892 | 0.897 |
| 2.5D CNN + series model | 0.915 (0.856–0.951) | 0.128 (0.102–0.159) | 0.946 (0.919–0.968) | 0.917 | 0.936 |

- The series model improves both CNNs by about +0.018 macro AUC and −0.011 log loss, and
  gains the most on the two subtypes the reference project named as weakest (subarachnoid
  +0.025, subdural +0.023 over the single-slice CNN).
- 2.5D input did **not** help at this data scale; the series model already supplies
  cross-slice context. The single-slice + series model is deployed.
- Differences between rows are smaller than the CIs; these are point estimates from one seed.

## Deployed chain (INT8 CNN → series model), certified through the serving code

`make export` runs the exported ONNX models through `serving.model.InferenceModel` on
validation (fit calibration and thresholds) and test (score once). ONNX vs PyTorch max logit
difference: 1.6e-5.

| Precision | Macro AUC | Weighted log loss | Study AUC (any) | CNN file |
|---|---|---|---|---|
| FP32 ONNX | 0.920 | 0.122 | 0.953 | 44.7 MB |
| **INT8 ONNX (deployed)** | **0.920 (0.854–0.957)** | **0.123 (0.099–0.154)** | **0.948 (0.918–0.971)** | **11.3 MB** |

Per label, INT8 deployed chain, test (slice level; thresholds F1-optimal on validation):

| Label | Positive slices | AUC (95% CI) | AP | ECE | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| any | 1,853 | 0.951 (0.936–0.964) | 0.840 | 0.011 | 0.796 | 0.932 |
| epidural | 39 | 0.759 (0.374–0.948) | 0.014 | 0.003 | 0.026 | 0.983 |
| intraparenchymal | 678 | 0.968 (0.954–0.978) | 0.782 | 0.005 | 0.664 | 0.985 |
| intraventricular | 460 | 0.974 (0.962–0.986) | 0.785 | 0.006 | 0.672 | 0.989 |
| subarachnoid | 621 | 0.929 (0.902–0.955) | 0.510 | 0.010 | 0.483 | 0.971 |
| subdural | 904 | 0.938 (0.916–0.960) | 0.613 | 0.009 | 0.636 | 0.949 |

Operating points for "any":

| Operating point (chosen on validation) | Level | Sensitivity | Specificity | PPV |
|---|---|---|---|---|
| Screening (≥95% sensitivity on validation) | slice | 0.970 | 0.698 | 0.396 |
| F1-optimal | study (max over slices) | 0.933 | 0.796 | 0.772 |

![ROC](figures/roc_test.png) ![Calibration](figures/calibration_test.png)

## Latency (CPU, 4 threads, M4)

`make benchmark` (`docs/benchmark.json`); 32-slice series, the chain runs the CNN in batches of 16.

| Engine | 1 slice p50 / p95 | 32-slice series p50 / p95 |
|---|---|---|
| PyTorch FP32 | 12.6 / 13.6 ms | 1,223 / 1,238 ms (CNN only) |
| ONNX Runtime FP32 | 18.8 / 20.7 ms | 809 / 833 ms (CNN + series model) |
| **ONNX Runtime INT8** | **8.5 / 9.7 ms** | **257 / 260 ms** (CNN + series model) |

INT8 is 3.1× faster than FP32 ONNX and 4.8× faster than PyTorch per series. At batch 1,
PyTorch on this CPU beat FP32 ONNX; INT8 is the fastest in every case.

## Serving on Kubernetes (kind on a 4-vCPU / 5 GB Colima VM)

**Correctness through the Service.** `scripts/e2e_real_series.py`: 60 real held-out series
(30 with hemorrhage) encoded as one DICOM per slice, uploaded in shuffled order through
`kubectl port-forward` → Service → pod. Served per-slice probabilities matched the offline
certified predictions within **5.0e-6**; study AUC 0.92 on this sample, identical to offline.
End-to-end latency (upload, decode, sort, inference) p50 410 ms, p95 550 ms per series.

**Autoscaling, in-cluster load** (`load_tests/job.yaml`, 32-slice synthetic series, closed
loop, 60 s steps, last 40 s measured). HPA: 1 → 3 pods within 31 s of load start.

| Concurrent clients | Series/s | p50 ms | p95 ms | Errors |
|---|---|---|---|---|
| 1 | 2.17 | 446 | 506 | 0 |
| 2 | 2.90 | 652 | 938 | 0 |
| 4 | 3.12 | 1,238 | 2,033 | 0 |
| 6 | 3.12 | 1,945 | 3,163 | 0 |
| 8 | 3.05 | 2,465 | 4,406 | 0 |

Throughput saturates at ~3.1 series/s (~100 slices/s) because three pods with 2-CPU limits
share the VM's 4 vCPUs; beyond that, more clients only add queueing. A Locust run through
port-forward (which pins one pod) plateaued at 2.0 series/s, which is why the in-cluster Job
is the measurement used here.

**Drift detection.** The load test's synthetic phantoms are not real head CT. Prometheus
recorded PSI of 7.3 (tissue mean HU), 8.5 (air fraction), 12.3 (brain-range fraction),
8.1 (bone fraction) and 1.6 (p(any)) against the training reference, and `BrainCTInputDrift`
went to pending (fires after 10 minutes above 0.2).

## CI (GitHub Actions, every push)

Lint, format, mypy, 66 tests with an 85% coverage gate (measured 94% of the service-side
code; 80 locally, where 9 PyTorch and 5 trained-bundle release tests also run), lockfile/requirements drift check; image build, Trivy scan (fixable critical CVEs
fail), no-PyTorch and non-root assertions, read-only-root container smoke test; kind cluster
+ Helm install, a real series request through the Service, rolling restart, monitoring
manifests applied.
