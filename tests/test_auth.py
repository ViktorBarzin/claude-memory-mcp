"""Tests for multi-user authentication."""

import importlib
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _reload_auth(env_vars: dict):
    """Reload the auth module with given environment variables."""
    with patch.dict(os.environ, env_vars, clear=False):
        # Clear existing env vars that might interfere
        for key in ("API_KEY", "API_KEYS"):
            os.environ.pop(key, None)
        for key, val in env_vars.items():
            os.environ[key] = val

        import claude_memory.api.auth as auth_mod

        importlib.reload(auth_mod)
        return auth_mod


@pytest.mark.asyncio
async def test_single_api_key_maps_to_default():
    auth = _reload_auth({"API_KEY": "test-key-123", "API_KEYS": ""})
    user = await auth.get_current_user(authorization="Bearer test-key-123")
    assert user.user_id == "default"


@pytest.mark.asyncio
async def test_multi_api_keys_maps_to_correct_user():
    auth = _reload_auth({
        "API_KEYS": '{"viktor": "key-viktor", "alice": "key-alice"}',
        "API_KEY": "",
    })
    user_v = await auth.get_current_user(authorization="Bearer key-viktor")
    assert user_v.user_id == "viktor"

    user_a = await auth.get_current_user(authorization="Bearer key-alice")
    assert user_a.user_id == "alice"


@pytest.mark.asyncio
async def test_invalid_key_returns_401():
    auth = _reload_auth({"API_KEY": "valid-key", "API_KEYS": ""})
    with pytest.raises(HTTPException) as exc_info:
        await auth.get_current_user(authorization="Bearer wrong-key")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_missing_bearer_prefix_still_works():
    auth = _reload_auth({"API_KEY": "my-key", "API_KEYS": ""})
    # Without Bearer prefix, removeprefix("Bearer ") returns "my-key" unchanged
    # so the raw token still matches the key
    user = await auth.get_current_user(authorization="my-key")
    assert user.user_id == "default"

    # With proper Bearer prefix it also works
    user = await auth.get_current_user(authorization="Bearer my-key")
    assert user.user_id == "default"


@pytest.mark.asyncio
async def test_missing_authorization_header_raises_422():
    """FastAPI raises 422 when required Header is missing.
    This is tested via the app integration, not the function directly,
    since FastAPI handles the missing header before the function runs.
    """
    from httpx import ASGITransport, AsyncClient

    # Need to reload with valid keys so the app can start
    _reload_auth({"API_KEY": "test-key", "API_KEYS": ""})

    # Import app after auth is configured
    import claude_memory.api.app as app_mod

    importlib.reload(app_mod)

    transport = ASGITransport(app=app_mod.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Skip lifespan since we don't have a real DB
        resp = await client.get("/api/memories")
        assert resp.status_code == 422
