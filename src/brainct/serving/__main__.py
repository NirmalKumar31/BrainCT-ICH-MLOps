"""uv run python -m brainct.serving [--host 0.0.0.0] [--port 8000]"""

from __future__ import annotations

import argparse

import uvicorn

from brainct.serving.app import create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    # One process per pod: scale with replicas (HPA), not uvicorn workers, so each pod's CPU
    # limit maps to one ONNX Runtime thread pool.
    uvicorn.run(create_app(), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
