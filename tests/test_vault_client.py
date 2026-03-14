"""Tests for Vault KV v2 client with mocked urllib."""

import json
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import pytest

from claude_memory.vault_client import VaultClient


@pytest.fixture
def vault_env(monkeypatch):
    monkeypatch.setenv("VAULT_ADDR", "http://vault.example.com:8200")
    monkeypatch.setenv("VAULT_TOKEN", "s.testtoken123")


class TestVaultClientInit:
    def test_missing_addr_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("VAULT_ADDR", raising=False)
        monkeypatch.delenv("VAULT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="Vault address not configured"):
            VaultClient()

    def test_init_with_explicit_args(self):
        client = VaultClient(addr="http://localhost:8200", token="mytoken")
        assert client.addr == "http://localhost:8200"
        assert client.token == "mytoken"
        assert client.mount == "secret"

    def test_init_from_env(self, vault_env):
        client = VaultClient()
        assert client.addr == "http://vault.example.com:8200"
        assert client.token == "s.testtoken123"

    def test_addr_trailing_slash_stripped(self):
        client = VaultClient(addr="http://localhost:8200/", token="t")
        assert client.addr == "http://localhost:8200"

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data="fake-jwt-token"))
    @patch("urllib.request.urlopen")
    def test_kubernetes_sa_token_auto_detection(self, mock_urlopen, mock_exists, monkeypatch):
        monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
        monkeypatch.delenv("VAULT_TOKEN", raising=False)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "auth": {"client_token": "s.k8s-token-abc"}
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = VaultClient()
        assert client.token == "s.k8s-token-abc"


class TestVaultRead:
    @patch("urllib.request.urlopen")
    def test_read_secret_returns_data(self, mock_urlopen, vault_env):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": {"data": {"username": "admin", "password": "secret"}}
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = VaultClient()
        result = client.read("myapp/config")
        assert result == {"username": "admin", "password": "secret"}

    @patch("urllib.request.urlopen")
    def test_read_returns_none_for_404(self, mock_urlopen, vault_env):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://vault:8200/v1/secret/data/missing",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=BytesIO(b""),
        )

        client = VaultClient()
        result = client.read("missing/path")
        assert result is None


class TestVaultWrite:
    @patch("urllib.request.urlopen")
    def test_write_secret_sends_correct_request(self, mock_urlopen, vault_env):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": {"created_time": "2024-01-01T00:00:00Z", "version": 1}
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = VaultClient()
        client.write("myapp/config", {"key": "value"})

        # Verify the request was made with correct data
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.full_url == "http://vault.example.com:8200/v1/secret/data/myapp/config"
        assert request.method == "POST"
        body = json.loads(request.data.decode())
        assert body == {"data": {"key": "value"}}


class TestVaultDelete:
    @patch("urllib.request.urlopen")
    def test_delete_returns_true_on_success(self, mock_urlopen, vault_env):
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = VaultClient()
        assert client.delete("myapp/config") is True

    @patch("urllib.request.urlopen")
    def test_delete_returns_false_on_error(self, mock_urlopen, vault_env):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://vault:8200/v1/secret/data/missing",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO(b"error"),
        )

        client = VaultClient()
        assert client.delete("missing/path") is False


class TestVaultListSecrets:
    @patch("urllib.request.urlopen")
    def test_list_secrets(self, mock_urlopen, vault_env):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": {"keys": ["secret1", "secret2/"]}
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = VaultClient()
        result = client.list_secrets("myapp")
        assert result == ["secret1", "secret2/"]
