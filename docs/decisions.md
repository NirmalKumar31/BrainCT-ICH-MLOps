# Design decisions

Each entry: what was chosen, the alternatives, and why. Numbers are from this repository's
runs (see `docs/results.md`).

## Data

**Source: RSNA 2019 as 16-bit PNG, streamed.** The original DICOM release is ~450 GB behind a
Kaggle login; PhysioNet's CT-ICH is now credentialed access. The 16-bit PNG conversion on
Hugging Face (`ianpan/rsna-intracranial-hemorrhage-16bit-png`, MIT, pinned by commit) keeps the
full dynamic range plus per-series rescale values, so HU are exact. It is one 144 GB tar, so
`brainct.data.ingest` opens 16 parallel HTTP range requests at evenly spaced offsets,
resynchronises on the first valid ustar header (checksum-verified), and converts complete
series straight to an int16 HU cache. Nothing raw is written. The tar stores series in hash
order, so each byte range is a pseudo-random sample of series. Result: 2,065 series / 71,736
slices in 141 s; series cut by a range start (13) are dropped.

**Sample size is bounded by disk, not bandwidth.** The laptop had ~20 GB free; at 224 px the
cache is ~100 KB per slice. ~10% of RSNA is enough to measure the architectural changes with
patient-level confidence intervals; it is not enough to match full-data leaderboard numbers.

**Patient-level split, stratified by each patient's rarest subtype.** Stratifying on "any"
alone left validation with zero epidural patients (0.5% of slices), so the epidural threshold
and calibration could not be fit. Strata are now the rarest subtype present
(epidural > intraventricular > subarachnoid > intraparenchymal > subdural > any > negative).
Result: 24 / 6 / 5 epidural patients in train / val / test.

## Model

**Tested 2.5D input; deployed single-slice CNN + series model.** The reference project
classified each slice alone and named that its main limitation (thin subdural and diffuse
subarachnoid bleeds are easier to see across slices). Two ways to add context were trained with
the same recipe and compared on the held-out test set (2×2 ablation, `docs/results.md`):
(a) 2.5D input, slices i-1, i, i+1 × 3 windows = 9 channels; (b) a series model over per-slice
embeddings. (b) improved macro AUC by ~0.018 for both CNNs; (a) did not help at this data
scale (0.897 vs 0.902 for the CNN alone, 0.915 vs 0.920 with the series model, CIs fully
overlapping). The series model already sees every slice, so the 2.5D channels were redundant.
The deployed bundle is therefore the single-slice CNN + series model: the better point
estimate and a simpler input contract. The code supports both (`context` in the config and
the model card); with the full 750k-slice dataset the 2.5D result may differ.

**Windowing inside the network graph.** The ONNX model takes HU; the window/clamp/normalise
ops are part of the graph. The service cannot window differently from training, and the
preprocessing contract shrinks to "HU, padded square, area-resized to 224". Those ops are kept
FP32 during INT8 quantization: uint8 over the raw HU range would be a 16 HU step, a fifth of
the 80 HU brain window.

**Series model: BiGRU over per-slice embeddings + CNN logits, residual on the logits.**
Initialised so it reproduces the CNN (zero output layer), it learns corrections from
whole-scan context: a bleed seen on slice 12 raises the prior on slices 11 and 13, isolated
false positives get damped. Trained in minutes on CPU. Known weakness: training embeddings come
from the CNN that saw those slices, so they are cleaner than test embeddings (the standard
fix is K-fold out-of-fold embeddings = K CNN trainings, out of budget here). Noise on the
input logits and embedding dropout narrow the gap; the gain is measured on held-out test only.

**Balanced sampling + post-hoc calibration.** Each epoch takes every positive slice and 1.25×
as many random negatives (~44% positive vs ~14% natural). This roughly halves epoch time
on a laptop GPU and helps the rare subtypes, but shifts predicted probabilities. Per-label
Platt scaling is fit on validation (of the *deployed* INT8 chain) and frozen; model selection
uses validation mean AUC, which the prior shift does not affect.

**ResNet-18 at 224 px, fp16 autocast on Apple MPS.** Measured on the M4: ~37 img/s sustained
training (9-channel 2.5D input; ~40 img/s for 3 channels), the fastest of ResNet-18/34, EfficientNet-B0 and ConvNeXt-nano;
bf16 was slower than fp16 on MPS. channels_last was slower.

## Evaluation

- Everything fit (calibration, thresholds) uses validation only; test is scored once.
- 95% CIs resample *patients* (slices of one patient are correlated).
- Reported: per-label ROC AUC and AP, the RSNA weighted log loss (with calibrated
  probabilities), ECE, study-level AUC for "any" (max over slices), and two operating points:
  F1-optimal and a 95%-sensitivity screening threshold for "any".
- The deployed artifact is what is certified: the INT8 CNN → series model chain runs through
  `serving.model.InferenceModel` on validation and test, and those are the headline numbers.

## Serving

- **Series endpoint first.** A study is a series, and the series model needs all slices.
  Slices are ordered by position along the slice normal (`ImagePositionPatient` ·
  orientation normal), then `InstanceNumber`, then upload order, with warnings. Mixed series
  are rejected. The single-slice endpoint remains (CNN head only, no series context, calibrated
  separately on validation); with a 2.5D bundle the neighbours are the slice itself, which
  training covered with context dropout (p=0.1).
- **No PyTorch in the image**; ONNX Runtime + numpy + OpenCV. Non-root, read-only root FS.
- **Bundle integrity:** the model card lists SHA-256 of every file; a mismatch keeps the pod
  un-ready (503 on `/ready`) instead of serving a wrong model.
- **Explainability:** class activation maps from the CNN feature map and head weights (one
  tensordot per slice, no gradient pass), returned as a PNG overlay on request.
- **Privacy:** uploads are processed in memory, never stored; logs carry request id, status
  and latency only; responses carry no DICOM tags.
- **Zip uploads** are checked for member count and *uncompressed* size before extraction.

## Monitoring

**Drift is monitored on the inputs and the outputs.** The reference project listed this as
the missing next step. Four cheap per-slice image features (mean tissue HU, air / brain-range /
bone fractions) catch scanner or protocol changes, wrong rescale tags and non-head images;
calibrated p(any) catches case-mix or model-behaviour shifts. The model card stores training
histograms; the service keeps a sliding window and exports PSI per feature. A Prometheus alert
fires on PSI > 0.2 for 10 minutes.

## Deployment

- Helm chart: HPA on CPU (scale up fast, down slowly), PodDisruptionBudget, startup /
  readiness / liveness probes on different endpoints, NetworkPolicy (ingress on 8000 only,
  no egress), no service-account token.
- CI deploys the chart to a kind cluster on every push and runs a real series request through
  the Service, then a rolling restart. The reference project deployed to GKE by hand; this
  repository targets kind/any cluster and verifies the deployment automatically, at no cost.
  No managed-cloud deployment was run for this version.
