# End-to-end pipeline. Each target is one stage; outputs are files, so stages can be rerun alone.
# Deployed run; cnn25d_resnet18 is the 2.5D ablation.
RUN      ?= cnn2d_resnet18
RUN_DIR  := artifacts/runs/$(RUN)
IMAGE    ?= brainct-api:0.2.0
PY       := uv run python

.PHONY: setup lint test data split train train-25d seq evaluate compare export benchmark \
        standin image serve smoke kind-up deploy monitoring loadtest kind-down

setup:            ## install everything (training + serving + dev)
	uv sync --all-extras --group loadtest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy

test:
	uv run pytest --cov

# ---------------------------------------------------------------- data + model
data:             ## stream ~2,000 RSNA series from Hugging Face into the HU cache (~7 GB)
	$(PY) -m brainct.data.ingest --out data/rsna --regions 16 --gib-per-region 0.8

split:
	$(PY) -m brainct.data.split --data data/rsna

train:            ## CNN (MPS/CUDA/CPU), then logits+embeddings for every split
	$(PY) -m brainct.training.train_cnn --config configs/$(RUN).yaml

train-25d:        ## 2.5D ablation (slices i-1, i, i+1)
	$(PY) -m brainct.training.train_cnn --config configs/cnn25d_resnet18.yaml
	$(PY) -m brainct.training.train_seq --config configs/cnn25d_resnet18.yaml

seq:              ## series BiGRU over stage-1 embeddings
	$(PY) -m brainct.training.train_seq --config configs/$(RUN).yaml

compare:          ## 2x2 ablation: (single slice | 2.5D) x (CNN | CNN + series model)
	$(PY) -m brainct.evaluation --compare artifacts/runs/cnn2d_resnet18:cnn \
	    artifacts/runs/cnn2d_resnet18:seq artifacts/runs/cnn25d_resnet18:cnn \
	    artifacts/runs/cnn25d_resnet18:seq

export:           ## ONNX + INT8 + certification of the deployed chain -> artifacts/bundle
	$(PY) -m brainct.optimization.export --run $(RUN_DIR) --out artifacts/bundle

benchmark:
	$(PY) -m brainct.optimization.benchmark --run $(RUN_DIR) --out docs/benchmark.json

# ---------------------------------------------------------------- serving
standin:          ## tiny model with the same interface (no RSNA weights)
	$(PY) scripts/make_standin_bundle.py artifacts/standin

image:            ## BUNDLE=artifacts/standin for a weights-free image
	docker build -f serving/Dockerfile --build-arg BUNDLE=$(or $(BUNDLE),artifacts/bundle) \
	    --build-arg GIT_SHA=$$(git rev-parse --short HEAD) -t $(IMAGE) .

serve:
	BRAINCT_BUNDLE=artifacts/bundle $(PY) -m brainct.serving

smoke:
	$(PY) scripts/smoke_test.py http://127.0.0.1:$(or $(PORT),8000) --wait 60

# ---------------------------------------------------------------- kubernetes (kind)
kind-up:
	kind create cluster --name brainct --wait 2m
	kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml
	kubectl -n kube-system patch deploy metrics-server --type=json \
	    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

deploy:
	kind load docker-image $(IMAGE) --name brainct
	helm upgrade --install brainct infra/helm/brainct -f infra/helm/brainct/values-kind.yaml --wait

monitoring:
	kubectl apply -k monitoring

loadtest:         ## needs: kubectl port-forward svc/brainct-brainct 8080:80
	mkdir -p load_tests/results
	uv run --group loadtest locust -f load_tests/locustfile.py --host http://127.0.0.1:8080 \
	    --headless --csv load_tests/results/run --html load_tests/results/run.html
	$(PY) scripts/summarize_locust.py load_tests/results/run_stats_history.csv

kind-down:
	kind delete cluster --name brainct
