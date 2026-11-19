import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Tests must not call AssemblyAI, Gemini, or an email provider."""

    def blocked(*_args, **_kwargs):
        raise AssertionError('network is disabled in tests')

    import requests

    monkeypatch.setattr(requests.sessions.Session, 'request', blocked)
