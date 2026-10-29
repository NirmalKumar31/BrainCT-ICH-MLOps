from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def _load_standin():
    spec = importlib.util.spec_from_file_location(
        "standin", ROOT / "scripts/make_standin_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def standin_bundle(tmp_path_factory) -> Path:
    return _load_standin().build(tmp_path_factory.mktemp("bundle"))


@pytest.fixture
def make_client(standin_bundle):
    """Factory: a TestClient (lifespan run) for a stand-in bundle and optional settings."""
    from fastapi.testclient import TestClient

    from brainct.serving.app import create_app
    from brainct.serving.settings import Settings

    clients = []

    def factory(**overrides) -> TestClient:
        settings = Settings(**({"bundle": str(standin_bundle), "drift_min_count": 5} | overrides))
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


REAL_BUNDLE = ROOT / "artifacts" / "bundle"
REAL_DATA = ROOT / "data" / "rsna"
