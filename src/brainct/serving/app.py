"""FastAPI service: upload a head CT series (DICOM files or a zip), get per-slice and study-level
hemorrhage probabilities. Research use only.

    uv run python -m brainct.serving            -> http://127.0.0.1:8000/docs

Uploads are processed in memory and never stored or logged; responses carry no DICOM tags.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)

from brainct import __version__
from brainct.constants import DISCLAIMER
from brainct.monitoring.drift import DriftMonitor, slice_features
from brainct.preprocessing import window
from brainct.serving.model import InferenceModel
from brainct.serving.schemas import (
    ErrorResponse,
    Heatmap,
    ModelRef,
    SeriesResponse,
    SlicePrediction,
    SliceResponse,
    StudySummary,
    Timing,
)
from brainct.serving.series import UploadError, Volume, assemble, decode_one, unpack_zip
from brainct.serving.settings import Settings

log = logging.getLogger("brainct.api")
ENDPOINTS = {
    "/v1/predict/series",
    "/v1/predict/slice",
    "/v1/model",
    "/v1/drift",
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/openapi.json",
}
LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    20.0,
)
ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse, "description": text}
    for code, text in {
        400: "nothing uploaded",
        413: "upload too large",
        422: "not a usable CT series",
        503: "model not loaded",
    }.items()
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def overlay_png(hu: np.ndarray, cam: np.ndarray) -> str:
    gray = (window(hu)[0] * 255).astype(np.uint8)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = (0.55 * cam)[..., None]
    mixed = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    ok, png = cv2.imencode(".png", mixed)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    log.handlers[:] = [handler]
    log.setLevel(settings.log_level)
    log.propagate = False

    registry = CollectorRegistry()
    ProcessCollector(registry=registry)  # CPU / RSS per pod (Linux /proc)
    m_requests = Counter(
        "brainct_requests", "HTTP requests", ["endpoint", "status"], registry=registry
    )
    m_latency = Histogram(
        "brainct_request_latency_seconds",
        "End-to-end latency",
        ["endpoint"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    m_stage = Histogram(
        "brainct_stage_seconds",
        "Time per stage",
        ["stage"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    m_slices = Histogram(
        "brainct_slices_per_request",
        "Slices per series request",
        buckets=(1, 8, 16, 24, 32, 40, 48, 64, 96, 128, 256),
        registry=registry,
    )
    m_positive = Counter(
        "brainct_positive_predictions",
        "Positive study-level calls by label",
        ["label"],
        registry=registry,
    )
    m_loaded = Gauge("brainct_model_loaded", "1 if the model bundle is loaded", registry=registry)
    m_info = Gauge(
        "brainct_model_info", "Loaded model version", ["name", "version"], registry=registry
    )
    m_inflight = Gauge("brainct_inflight_inferences", "Model calls running", registry=registry)
    m_psi = Gauge(
        "brainct_drift_psi", "PSI of live vs training distribution", ["feature"], registry=registry
    )
    m_drift_n = Gauge(
        "brainct_drift_window_count", "Values in the drift window", ["feature"], registry=registry
    )

    state: dict[str, Any] = {"model": None, "error": None, "drift": None}
    gate = asyncio.Semaphore(settings.max_concurrency)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # A failed load leaves /health up and /ready at 503, so Kubernetes can tell
        # "process alive" from "able to serve" and keeps the pod out of the Service.
        try:
            model = InferenceModel(settings.bundle, settings.threads, settings.verify_bundle)
            if settings.warmup:
                size = model.image_size
                model.run(np.zeros((4, size, size), dtype=np.int16), with_fmap=True)
            state["model"] = model
            state["drift"] = DriftMonitor(
                model.card.get("drift_reference", {}),
                settings.drift_window,
                settings.drift_min_count,
            )
            m_loaded.set(1)
            m_info.labels(model.name, model.version).set(1)
            log.info("model loaded", extra={"fields": {"version": model.version}})
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            m_loaded.set(0)
            log.error("model load failed", extra={"fields": {"error": state["error"]}})
        yield

    app = FastAPI(
        title="BrainCT intracranial hemorrhage detection (research use only)",
        description=DISCLAIMER,
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        response = await call_next(request)
        path = request.url.path
        endpoint = path if path in ENDPOINTS else "other"  # bounded label cardinality
        elapsed = time.perf_counter() - start
        m_requests.labels(endpoint, str(response.status_code)).inc()
        m_latency.labels(endpoint).observe(elapsed)
        response.headers["x-request-id"] = rid
        if endpoint.startswith("/v1/predict"):
            log.info(
                "request",
                extra={
                    "fields": {
                        "rid": rid,
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "ms": round(elapsed * 1000, 1),
                    }
                },
            )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(ErrorResponse(detail=str(exc.detail)).model_dump(), exc.status_code)

    def require_model() -> InferenceModel:
        if state["model"] is None:
            raise HTTPException(503, f"model not loaded: {state['error']}")
        return state["model"]

    async def read_upload(f: UploadFile, limit: int) -> bytes:
        data = await f.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"{f.filename or 'file'} exceeds {limit // 2**20} MB")
        return data

    async def infer(fn, *args):
        async with gate:
            m_inflight.inc()
            try:
                return await run_in_threadpool(fn, *args)
            finally:
                m_inflight.dec()

    def slice_prediction(model, probs, head, i, info) -> SlicePrediction:
        thr = model.thresholds(head)
        return SlicePrediction(
            index=i,
            instance_number=info.instance_number,
            z=info.z,
            probabilities={k: round(float(p), 5) for k, p in zip(model.labels, probs, strict=True)},
            positive=[k for k, p in zip(model.labels, probs, strict=True) if p >= thr[k]],
        )

    def heatmap(model, vol: np.ndarray, fmap: np.ndarray, i: int, label: str) -> Heatmap:
        cam = model.cam(fmap[i], model.labels.index(label))
        return Heatmap(slice_index=i, label=label, png_base64=overlay_png(vol[i], cam))

    def observe_drift(vol: np.ndarray, p_any: np.ndarray) -> None:
        drift: DriftMonitor = state["drift"]
        for i in range(len(vol)):
            drift.observe(slice_features(vol[i]) | {"p_any": float(p_any[i])})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"model": ErrorResponse}})
    async def ready() -> dict[str, str]:
        model = require_model()
        return {"status": "ready", "model": model.name, "version": model.version}

    @app.get("/metrics")
    async def metrics() -> Response:
        if state["drift"] is not None:
            for name, s in state["drift"].snapshot().items():
                m_drift_n.labels(name).set(s["count"])
                if s["psi"] is not None:
                    m_psi.labels(name).set(s["psi"])
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model")
    async def model_card() -> dict[str, Any]:
        model = require_model()
        card = model.card
        return {
            "name": card["name"],
            "version": card["version"],
            "contract": card["contract"],
            "thresholds": card["thresholds"],
            "metrics_test": card.get("metrics_test"),
            "intended_use": card.get("intended_use"),
            "created": card.get("provenance", {}).get("created"),
            "git_sha": card.get("provenance", {}).get("git_sha"),
        }

    @app.get("/v1/drift")
    async def drift() -> dict[str, Any]:
        require_model()
        return {"window": settings.drift_window, "features": state["drift"].snapshot()}

    @app.post("/v1/predict/series", response_model=SeriesResponse, responses=ERRORS)
    async def predict_series(
        files: Annotated[
            list[UploadFile], File(description="DICOM slices of one series, or one .zip")
        ],
        explain: Annotated[bool, Query(description="add a CAM heatmap of the key slice")] = False,
    ) -> SeriesResponse:
        t0 = time.perf_counter()
        model = require_model()
        file_limit = int(settings.max_file_mb * 2**20)
        total_limit = int(settings.max_total_mb * 2**20)
        blobs: list[bytes] = []
        total = 0
        try:
            for f in files:
                is_zip = (f.filename or "").lower().endswith(".zip")
                data = await read_upload(f, total_limit if is_zip else file_limit)
                if is_zip:
                    blobs.extend(
                        await run_in_threadpool(unpack_zip, data, settings.max_slices, total_limit)
                    )
                else:
                    blobs.append(data)
                total += len(data)
                if total > total_limit or len(blobs) > settings.max_slices:
                    raise UploadError(
                        413,
                        f"limit is {settings.max_slices} slices / "
                        f"{settings.max_total_mb:.0f} MB per request",
                    )
            if not blobs or not any(blobs):
                raise UploadError(400, "no slices uploaded")

            def decode() -> Volume:
                return assemble([decode_one(b, model.image_size) for b in blobs])

            vol = await run_in_threadpool(decode)
        except UploadError as exc:
            raise HTTPException(exc.status, exc.detail) from exc
        t1 = time.perf_counter()
        raw = await infer(model.run, vol.hu, True, explain)
        t2 = time.perf_counter()

        probs = model.calibrate(raw.seq_logits, "seq")
        thr = model.thresholds("seq")
        slices = [
            slice_prediction(model, probs[i], "seq", i, vol.infos[i]) for i in range(len(probs))
        ]
        study_p = probs.max(axis=0)
        key = int(np.argmax(probs[:, 0]))
        subtypes = [k for j, k in enumerate(model.labels) if j and study_p[j] >= thr[k]]
        study = StudySummary(
            hemorrhage=bool(study_p[0] >= thr["study_any"]),
            screening_positive=bool(study_p[0] >= thr["screening_any"]),
            probabilities={
                k: round(float(p), 5) for k, p in zip(model.labels, study_p, strict=True)
            },
            subtypes=subtypes,
            key_slice=key,
        )
        if study.hemorrhage:
            m_positive.labels("any").inc()
        for k in subtypes:
            m_positive.labels(k).inc()
        observe_drift(vol.hu, probs[:, 0])
        hm = None
        if explain and raw.fmap is not None:
            label = (
                max(subtypes, key=lambda k: study_p[model.labels.index(k)]) if subtypes else "any"
            )
            hm = heatmap(model, vol.hu, raw.fmap, key, label)
        m_stage.labels("decode").observe(t1 - t0)
        m_stage.labels("inference").observe(t2 - t1)
        m_slices.observe(len(vol.hu))
        return SeriesResponse(
            model=ModelRef(name=model.name, version=model.version, head="seq"),
            study=study,
            slices=slices,
            ordering=vol.order,
            warnings=vol.warnings,
            heatmap=hm,
            timing=Timing(
                decode_ms=(t1 - t0) * 1e3,
                inference_ms=(t2 - t1) * 1e3,
                total_ms=(time.perf_counter() - t0) * 1e3,
            ),
        )

    @app.post("/v1/predict/slice", response_model=SliceResponse, responses=ERRORS)
    async def predict_slice(
        file: Annotated[UploadFile, File(description="one CT slice as DICOM")],
        explain: Annotated[bool, Query(description="add a CAM heatmap")] = False,
    ) -> SliceResponse:
        t0 = time.perf_counter()
        model = require_model()
        data = await read_upload(file, int(settings.max_file_mb * 2**20))
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            hu, info = await run_in_threadpool(decode_one, data, model.image_size)
        except UploadError as exc:
            raise HTTPException(exc.status, exc.detail) from exc
        t1 = time.perf_counter()
        raw = await infer(model.run, hu[None], False, explain)
        t2 = time.perf_counter()
        probs = model.calibrate(raw.cnn_logits, "cnn")
        observe_drift(hu[None], probs[:, 0])
        pred = slice_prediction(model, probs[0], "cnn", 0, info)
        hm = None
        if explain and raw.fmap is not None:
            label = max(pred.positive[1:] or ["any"], key=lambda k: pred.probabilities[k])
            hm = heatmap(model, hu[None], raw.fmap, 0, label)
        m_stage.labels("decode").observe(t1 - t0)
        m_stage.labels("inference").observe(t2 - t1)
        warnings = ["single slice: no neighbouring context; prefer /v1/predict/series"]
        if info.sign_fixed:
            warnings.append("corrected 12-bit sign encoding")
        return SliceResponse(
            model=ModelRef(name=model.name, version=model.version, head="cnn"),
            prediction=pred,
            warnings=warnings,
            heatmap=hm,
            timing=Timing(
                decode_ms=(t1 - t0) * 1e3,
                inference_ms=(t2 - t1) * 1e3,
                total_ms=(time.perf_counter() - t0) * 1e3,
            ),
        )

    return app
