# syntax=docker/dockerfile:1.7
# BrainCT inference API: ONNX Runtime + numpy, no PyTorch. Research use only.
#
#   docker build -f serving/Dockerfile -t brainct-api .                         # real bundle
#   docker build -f serving/Dockerfile --build-arg BUNDLE=artifacts/standin -t brainct-api:ci .
#
# serving/requirements.txt is exported from uv.lock (CI fails if it drifts):
#   uv export --frozen --no-dev --extra serve --no-emit-project --no-hashes -o serving/requirements.txt

FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY serving/requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
ARG BUNDLE=artifacts/bundle
ARG VERSION=dev
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="brainct-api" \
      org.opencontainers.image.description="Intracranial hemorrhage detection API (research use only)" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="MIT"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BRAINCT_BUNDLE=/app/bundle
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=deps /opt/venv /opt/venv
COPY src/brainct /app/brainct
COPY ${BUNDLE} /app/bundle
USER 10001
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"]
CMD ["python", "-m", "brainct.serving", "--host", "0.0.0.0", "--port", "8000"]
