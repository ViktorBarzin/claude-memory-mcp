"""HashiCorp Vault KV v2 client using stdlib urllib."""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class VaultClient:
    """Simple Vault KV v2 client using stdlib."""

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        mount: str = "secret",
    ):
        self.addr = (addr or os.environ.get("VAULT_ADDR", "")).rstrip("/")
        self.token = token or os.environ.get("VAULT_TOKEN", "")
        self.mount = mount

        if not self.addr:
            raise ValueError("Vault address not configured (set VAULT_ADDR)")

        # Auto-detect Kubernetes SA token
        if not self.token:
            sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            if os.path.exists(sa_token_path):
                self._login_kubernetes(sa_token_path)

    def _login_kubernetes(self, sa_token_path: str) -> None:
        """Authenticate with Vault using Kubernetes service account."""
        with open(sa_token_path) as f:
            jwt = f.read().strip()
        role = os.environ.get("VAULT_ROLE", "claude-memory")
        resp = self._request("POST", "/v1/auth/kubernetes/login", {"jwt": jwt, "role": role})
        self.token = resp.get("auth", {}).get("client_token", "")

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make HTTP request to Vault."""
        url = f"{self.addr}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"X-Vault-Token": self.token, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result: dict[str, Any] = json.loads(resp.read().decode())
                return result
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            error_body = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"Vault error {e.code}: {error_body}") from e

    def read(self, path: str) -> dict[str, Any] | None:
        """Read a secret from KV v2."""
        resp = self._request("GET", f"/v1/{self.mount}/data/{path}")
        data = resp.get("data", {})
        return data.get("data") if data else None

    def write(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Write a secret to KV v2."""
        return self._request("POST", f"/v1/{self.mount}/data/{path}", {"data": data})

    def delete(self, path: str) -> bool:
        """Delete a secret from KV v2."""
        try:
            self._request("DELETE", f"/v1/{self.mount}/data/{path}")
            return True
        except RuntimeError:
            return False

    def list_secrets(self, path: str) -> list[str]:
        """List secrets at a path."""
        try:
            resp = self._request("LIST", f"/v1/{self.mount}/metadata/{path}")
            keys: list[str] = resp.get("data", {}).get("keys", [])
            return keys
        except RuntimeError:
            return []
