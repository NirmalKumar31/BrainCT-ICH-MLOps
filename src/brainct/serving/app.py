"""FastAPI service: upload a head CT series (DICOM files or a zip), get per-slice and study-level
hemorrhage probabilities. Research use only.

    uv run python -m brainct.serving            -> http://127.0.0.1:8000/docs

Uploads are processed in memory and never stored or logged; responses carry no DICOM tags.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
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
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from brainct import __version__
from brainct.constants import DISCLAIMER
from brainct.monitoring.drift import DriftMonitor, slice_features
from brainct.monitoring.performance import PerformanceMonitor
from brainct.preprocessing import window
from brainct.serving.batching import MicroBatcher
from brainct.serving.model import BundleError, InferenceModel, RawOutput, contexts
from brainct.serving.schemas import (
    ErrorResponse,
    Feedback,
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
    "/v1/feedback",
    "/v1/performance",
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


CLIENT_RID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


class BodyTooLarge(Exception):
    pass


class BodyLimit:
    """Reject request bodies over `limit` bytes while they stream in.

    The per-file and per-series checks in the handlers run only after Starlette has parsed the
    whole multipart body (spooling parts > 1 MB to /tmp); without this, one request can fill
    memory or the /tmp volume before any limit is looked at.
    """

    def __init__(self, app: ASGIApp, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            await self._reject(send)
            return
        seen = 0

        async def limited() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise BodyTooLarge
            return message

        await self.app(scope, limited, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps({"detail": f"request body exceeds {self.limit // 2**20} MB"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


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
    m_precision = Gauge(
        "brainct_precision",
        "CNN precision serving (self-test may force fp32)",
        ["precision"],
        registry=registry,
    )
    m_inflight = Gauge("brainct_inflight_inferences", "Model calls running", registry=registry)
    m_psi = Gauge(
        "brainct_drift_psi", "PSI of live vs training distribution", ["feature"], registry=registry
    )
    m_drift_n = Gauge(
        "brainct_drift_window_count", "Values in the drift window", ["feature"], registry=registry
    )

    m_batch = Histogram(
        "brainct_batch_slices",
        "Slices per micro-batched CNN call",
        buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64),
        registry=registry,
    )
    m_feedback = Counter(
        "brainct_feedback", "Ground-truth feedback received", ["result"], registry=registry
    )
    m_live = Gauge(
        "brainct_live_metric",
        "Study-level quality on labelled feedback (sliding window)",
        ["label", "metric"],
        registry=registry,
    )

    state: dict[str, Any] = {
        "model": None,
        "error": None,
        "public_error": None,
        "drift": None,
        "batcher": None,
        "perf": None,
    }
    gate = asyncio.Semaphore(settings.max_concurrency)  # model calls
    decode_gate = asyncio.Semaphore(settings.max_decodes)  # decoded series held in memory

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # A failed load leaves /health up and /ready at 503, so Kubernetes can tell
        # "process alive" from "able to serve" and keeps the pod out of the Service.
        try:
            model = InferenceModel(
                settings.bundle, settings.threads, settings.verify_bundle, settings.precision
            )
            if settings.warmup:
                size = model.image_size
                model.run(np.zeros((4, size, size), dtype=np.int16), with_fmap=True)
            state["model"] = model
            state["drift"] = DriftMonitor(
                model.card.get("drift_reference", {}),
                settings.drift_window,
                settings.drift_min_count,
            )
            state["perf"] = PerformanceMonitor(model.labels, window=settings.feedback_window)
            if settings.batching:
                state["batcher"] = MicroBatcher(
                    model.cnn_forward, settings.max_batch, settings.max_wait_ms, m_batch.observe
                )
                await state["batcher"].start()
            m_loaded.set(1)
            m_info.labels(model.name, model.version).set(1)
            m_precision.labels(model.precision).set(1)
            level = logging.WARNING if model.selftest.get("passed") is False else logging.INFO
            fields = {
                "version": model.version,
                "precision": model.precision,
                "selftest": model.selftest,
            }
            log.log(level, "model loaded", extra={"fields": fields})
        except Exception as exc:
            # Details go to the log only; clients see a generic 503.
            state["error"] = f"{type(exc).__name__}: {exc}"
            # BundleError messages name the problem without paths; anything else stays generic.
            state["public_error"] = str(exc) if isinstance(exc, BundleError) else "internal error"
            m_loaded.set(0)
            log.error("model load failed", extra={"fields": {"error": state["error"]}})
        yield
        if state["batcher"] is not None:
            await state["batcher"].stop()

    app = FastAPI(
        title="BrainCT intracranial hemorrhage detection (research use only)",
        description=DISCLAIMER,
        version=__version__,
        lifespan=lifespan,
    )

    # Starlette puts each newly added middleware outermost, so `observe` (added next) wraps
    # this one: its 413s are counted, and BodyTooLarge raised mid-stream reaches `observe`.
    app.add_middleware(BodyLimit, limit=int(settings.max_request_mb * 2**20))

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        # Server-generated, so it is unique: it is the key /v1/feedback joins on. A client id
        # (if well-formed) is only logged, for tracing.
        rid = uuid.uuid4().hex[:16]
        client_rid = request.headers.get("x-request-id", "")
        request.state.rid = rid
        try:
            response = await call_next(request)
        except BodyTooLarge:
            response = JSONResponse(
                {"detail": f"request body exceeds {settings.max_request_mb:.0f} MB"}, 413
            )
        except Exception:
            # Unhandled errors must still be counted (the 5xx alert) and carry the id.
            log.exception("unhandled error", extra={"fields": {"rid": rid}})
            response = JSONResponse({"detail": "internal error", "request_id": rid}, 500)
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
                        **({"client_rid": client_rid} if CLIENT_RID.fullmatch(client_rid) else {}),
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
            raise HTTPException(503, f"model not loaded: {state['public_error']}")
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

    async def run_model(model: InferenceModel, vol: np.ndarray, with_seq: bool, explain: bool):
        """CNN (+ series model). Explain needs the feature map, so it bypasses the batcher."""
        batcher: MicroBatcher | None = state["batcher"]
        if batcher is None or explain:
            return await infer(model.run, vol, with_seq, explain)
        x = await run_in_threadpool(contexts, vol, model.context)
        logits, emb = await batcher.submit(x)
        seq = await infer(model.seq_forward, emb, logits) if with_seq else None
        return RawOutput(logits, seq, None)

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

    def observe_drift(vol: np.ndarray, p_any: np.ndarray | None) -> None:
        """Input features always; p_any only from the head the reference was built on (seq)."""
        drift: DriftMonitor = state["drift"]
        for i in range(len(vol)):
            feats = slice_features(vol[i])
            drift.observe(feats if p_any is None else feats | {"p_any": float(p_any[i])})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"model": ErrorResponse}})
    async def ready() -> dict[str, str]:
        model = require_model()
        return {
            "status": "ready",
            "model": model.name,
            "version": model.version,
            "precision": model.precision,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        if state["drift"] is not None:
            for label, stats in state["perf"].snapshot().items():
                for metric in ("auc", "brier", "ece", "count"):
                    if stats[metric] is not None:
                        m_live.labels(label, metric).set(stats[metric])
                    else:  # undefined now (e.g. one class in the window): drop, don't go stale
                        with contextlib.suppress(KeyError):
                            m_live.remove(label, metric)
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
            "precision": model.precision,
            "selftest": model.selftest,
            "created": card.get("provenance", {}).get("created"),
            "git_sha": card.get("provenance", {}).get("git_sha"),
        }

    @app.post(
        "/v1/feedback",
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    async def feedback(
        body: Feedback,
        x_feedback_token: Annotated[str, Header()] = "",
    ) -> dict[str, Any]:
        """Ground truth for an earlier series prediction (study level), by its x-request-id."""
        require_model()
        if settings.feedback_token and not secrets.compare_digest(
            x_feedback_token, settings.feedback_token
        ):
            m_feedback.labels("unauthorized").inc()
            raise HTTPException(401, "missing or wrong x-feedback-token")
        try:
            joined = state["perf"].feedback(body.request_id, body.labels)
        except ValueError as exc:
            m_feedback.labels("invalid").inc()
            raise HTTPException(422, str(exc)) from exc
        if not joined:
            m_feedback.labels("unknown_request").inc()
            raise HTTPException(404, "unknown or expired request_id")
        m_feedback.labels("joined").inc()
        return {"status": "recorded"}

    @app.get("/v1/performance")
    async def performance() -> dict[str, Any]:
        require_model()
        return {"window": settings.feedback_window, "labels": state["perf"].snapshot()}

    @app.get("/v1/drift")
    async def drift() -> dict[str, Any]:
        require_model()
        return {"window": settings.drift_window, "features": state["drift"].snapshot()}

    @app.post("/v1/predict/series", response_model=SeriesResponse, responses=ERRORS)
    async def predict_series(
        request: Request,
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
        total = 0  # uncompressed bytes: what decoding will actually hold in memory
        too_much = UploadError(
            413,
            f"limit is {settings.max_slices} slices / {settings.max_total_mb:.0f} MB per request",
        )
        async with decode_gate:
            try:
                for f in files:
                    is_zip = (f.filename or "").lower().endswith(".zip")
                    data = await read_upload(f, total_limit if is_zip else file_limit)
                    if is_zip:
                        new = await run_in_threadpool(
                            unpack_zip,
                            data,
                            settings.max_slices - len(blobs),
                            total_limit - total,
                            file_limit,
                        )
                    else:
                        new = [data]
                    blobs.extend(new)
                    total += sum(len(b) for b in new)
                    if total > total_limit or len(blobs) > settings.max_slices:
                        raise too_much
                if not blobs or not any(blobs):
                    raise UploadError(400, "no slices uploaded")

                def decode() -> Volume:
                    return assemble([decode_one(b, model.image_size) for b in blobs])

                vol = await run_in_threadpool(decode)
            except UploadError as exc:
                raise HTTPException(exc.status, exc.detail) from exc
            blobs.clear()  # free the raw bytes before inference
            t1 = time.perf_counter()
            head = "seq" if model.has_seq else "cnn"  # a bundle without a series model
            raw = await run_model(model, vol.hu, head == "seq", explain)
        t2 = time.perf_counter()

        logits = raw.seq_logits if head == "seq" else raw.cnn_logits
        assert logits is not None
        probs = model.calibrate(logits, head)
        thr = model.study_thresholds(head)
        slices = [
            slice_prediction(model, probs[i], head, i, vol.infos[i]) for i in range(len(probs))
        ]
        study_p = probs.max(axis=0)
        key = int(np.argmax(probs[:, 0]))
        subtypes = [k for j, k in enumerate(model.labels) if j and study_p[j] >= thr[k]]
        study = StudySummary(
            hemorrhage=bool(study_p[0] >= thr["any"]),
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
        await run_in_threadpool(observe_drift, vol.hu, probs[:, 0] if head == "seq" else None)
        state["perf"].record(request.state.rid, study_p)
        hm = None
        if explain and raw.fmap is not None:
            label = (
                max(subtypes, key=lambda k: study_p[model.labels.index(k)]) if subtypes else "any"
            )
            hm = await run_in_threadpool(heatmap, model, vol.hu, raw.fmap, key, label)
        m_stage.labels("decode").observe(t1 - t0)
        m_stage.labels("inference").observe(t2 - t1)
        m_slices.observe(len(vol.hu))
        return SeriesResponse(
            model=ModelRef(name=model.name, version=model.version, head=head),
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
            async with decode_gate:
                hu, info = await run_in_threadpool(decode_one, data, model.image_size)
        except UploadError as exc:
            raise HTTPException(exc.status, exc.detail) from exc
        t1 = time.perf_counter()
        raw = await run_model(model, hu[None], False, explain)
        t2 = time.perf_counter()
        probs = model.calibrate(raw.cnn_logits, "cnn")
        observe_drift(hu[None], None)  # CNN-head probabilities are not the drift reference's
        pred = slice_prediction(model, probs[0], "cnn", 0, info)
        hm = None
        if explain and raw.fmap is not None:
            label = max(pred.positive[1:] or ["any"], key=lambda k: pred.probabilities[k])
            hm = await run_in_threadpool(heatmap, model, hu[None], raw.fmap, 0, label)
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
