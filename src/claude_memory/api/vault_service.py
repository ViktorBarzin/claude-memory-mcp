import os
import logging

logger = logging.getLogger(__name__)

VAULT_ADDR = os.environ.get("VAULT_ADDR", "")
VAULT_TOKEN = os.environ.get("VAULT_TOKEN", "")
VAULT_MOUNT = os.environ.get("VAULT_MOUNT", "secret")
VAULT_PREFIX = os.environ.get("VAULT_PREFIX", "claude-memory")


def is_vault_configured() -> bool:
    return bool(VAULT_ADDR and VAULT_TOKEN)


async def store_secret(user_id: str, memory_id: int, content: str) -> str:
    """Store secret content in Vault. Returns the vault path."""
    if not is_vault_configured():
        raise RuntimeError("Vault not configured")

    from claude_memory.vault_client import VaultClient

    client = VaultClient(VAULT_ADDR, VAULT_TOKEN, VAULT_MOUNT)
    path = f"{VAULT_PREFIX}/{user_id}/mem-{memory_id}"
    client.write(path, {"content": content})
    return path


async def get_secret(user_id: str, vault_path: str) -> str | None:
    """Retrieve secret content from Vault."""
    if not is_vault_configured():
        return None

    from claude_memory.vault_client import VaultClient

    client = VaultClient(VAULT_ADDR, VAULT_TOKEN, VAULT_MOUNT)
    data = client.read(vault_path)
    if data:
        return data.get("content")
    return None


async def delete_secret(user_id: str, vault_path: str) -> bool:
    """Delete secret from Vault."""
    if not is_vault_configured():
        return False

    from claude_memory.vault_client import VaultClient

    client = VaultClient(VAULT_ADDR, VAULT_TOKEN, VAULT_MOUNT)
    return client.delete(vault_path)
