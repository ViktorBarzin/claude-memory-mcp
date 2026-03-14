import json
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException


@dataclass
class AuthUser:
    user_id: str


# Multi-user mode: API_KEYS='{"viktor": "key1", "user2": "key2"}'
# Single-user mode: API_KEY="some-key" (backward compatible, user_id="default")
_api_keys_json = os.environ.get("API_KEYS", "")
_api_key_single = os.environ.get("API_KEY", "")

_key_to_user: dict[str, str] = {}

if _api_keys_json:
    _user_to_key = json.loads(_api_keys_json)
    _key_to_user = {v: k for k, v in _user_to_key.items()}
elif _api_key_single:
    _key_to_user = {_api_key_single: "default"}


async def get_current_user(authorization: str = Header(...)) -> AuthUser:
    token = authorization.removeprefix("Bearer ").strip()
    user_id = _key_to_user.get(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return AuthUser(user_id=user_id)
