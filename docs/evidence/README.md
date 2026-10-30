# Evidence

Raw outputs behind the numbers in [`../results.md`](../results.md). Images contain synthetic
phantoms or dashboards only; no patient image is in this repository.

## Model and training

| File | What it shows |
|---|---|
| [`modal_full_data_training_curves.png`](modal_full_data_training_curves.png), [`modal_full_data_curves.json`](modal_full_data_curves.json) | Full-data (all of RSNA) ResNet-50 runs on a Modal A100: per-epoch validation AUC and series-model log loss |
| [`modal/`](modal/) | Logs of the full-data pipeline: ingest (753 s, 21,744 series), both trainings (`nvidia-smi`, throughput), series model, export and certification |
| [`full_data_certification.json`](full_data_certification.json), [`full_data_model_card.json`](full_data_model_card.json) | Deployed full-data chain on the test set (FP32 and INT8, per label, CIs) and the served model card |
| [`full_data_e2e_real_series_local.json`](full_data_e2e_real_series_local.json), [`full_data_live_performance_local.json`](full_data_live_performance_local.json) | 60 real held-out series through the API (x86-certified, ARM-served) and the live metrics from their feedback |
| [`full_data_benchmark_modal_cpu.json`](full_data_benchmark_modal_cpu.json), [`../benchmark_full_resnet50.json`](../benchmark_full_resnet50.json) | Latency on Modal x86 and on the M4 |
| [`../figures/full/`](../figures/full/) | Test ROC and calibration of the deployed full-data chain |
| [`mlflow_training_curves.png`](mlflow_training_curves.png), [`mlflow_runs.json`](mlflow_runs.json) | Laptop-sample runs (ResNet-18) from the MLflow store: per-epoch validation AUC (CNNs) and log loss (series models) |
| [`certification_summary.json`](certification_summary.json) | Laptop-sample chain test metrics, FP32 vs INT8, from `make export` |
| [`../figures/roc_test.png`](../figures/roc_test.png), [`../figures/calibration_test.png`](../figures/calibration_test.png) | Test ROC and calibration of the laptop-sample chain |
| [`ddp_gloo_2proc.log`](ddp_gloo_2proc.log), [`ddp_gloo_2proc_uneven.log`](ddp_gloo_2proc_uneven.log) | Distributed (DDP) training with 2 processes; and an epoch size that used to deadlock (audit fix) |
| [`quant_variants_x86_epyc_milan.json`](quant_variants_x86_epyc_milan.json), [`quant_variants_arm_apple_m4.json`](quant_variants_arm_apple_m4.json) | INT8 schemes measured on the GKE x86 CPU and on the M4 (phantom + real validation slices; the incident in `../decisions.md`) |
| [`int8_reduce_range_rejected.log`](int8_reduce_range_rejected.log) | The rejected reduce_range INT8 export: series-model chain macro AUC 0.843, log loss 2.21 |

## Service

| File | What it shows |
|---|---|
| [`swagger_ui.png`](swagger_ui.png) | OpenAPI docs of the running service |
| [`example_series_response.json`](example_series_response.json) | A real response of the trained model (synthetic phantom input, excerpt) |
| [`heatmap_synthetic_phantom.png`](heatmap_synthetic_phantom.png) | The CAM heatmap returned with `explain=true` |
| [`example_errors.json`](example_errors.json) | 4xx responses for bad uploads |
| [`v1_model.json`](v1_model.json), [`v1_drift.json`](v1_drift.json) | Model card and drift state served by the API (laptop ResNet-18 bundle) |
| [`pytest_coverage.txt`](pytest_coverage.txt), [`ci_runs.txt`](ci_runs.txt) | Local test run with coverage; CI runs at the time of writing ([live](https://github.com/NirmalKumar31/BrainCT-ICH-MLOps/actions)) |

## Kubernetes, local (kind)

| File | What it shows |
|---|---|
| [`kind_load_steps.md`](kind_load_steps.md), [`kind_hpa_timeline.txt`](kind_hpa_timeline.txt) | In-cluster step load and HPA scaling on a 4-vCPU VM |

## Kubernetes, cloud (GKE Autopilot, ResNet-18 bundle)

| File | What it shows |
|---|---|
| [`gke/terraform_apply_summary.txt`](gke/terraform_apply_summary.txt), [`gke/terraform_destroy_summary.txt`](gke/terraform_destroy_summary.txt) | Infrastructure created and destroyed by Terraform |
| [`gke/kubectl_state.txt`](gke/kubectl_state.txt) | Nodes, Deployment, HPA, PDB, NetworkPolicy, pods, HPA events |
| [`gke/model_card_served.json`](gke/model_card_served.json) | Version, precision and startup self-test result of the pod on the EPYC CPU |
| [`gke/e2e_real_series.json`](gke/e2e_real_series.json) | 60 real series through the cloud Service vs the certified offline predictions |
| [`gke/load_steps.md`](gke/load_steps.md), [`gke/hpa_timeline.txt`](gke/hpa_timeline.txt) | Autoscaling under in-cluster load |
| [`gke/grafana_dashboard_under_load.png`](gke/grafana_dashboard_under_load.png) | Grafana during the load test |
| [`gke/grafana_dashboard_after_feedback.png`](gke/grafana_dashboard_after_feedback.png) | Grafana after ground-truth feedback (live AUC panel) |
| [`gke/prometheus_alerts.png`](gke/prometheus_alerts.png), [`gke/alertmanager_received.txt`](gke/alertmanager_received.txt) | Drift and latency alerts firing and delivered to Alertmanager |
| [`gke/prometheus_targets.png`](gke/prometheus_targets.png) | Prometheus discovering and scraping every API pod |
| [`gke/live_performance_feedback.json`](gke/live_performance_feedback.json) | Live per-label AUC / Brier / ECE from 80 labelled series |
