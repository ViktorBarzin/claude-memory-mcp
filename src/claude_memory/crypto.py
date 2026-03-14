"""AES-256-GCM encryption for memory content when Vault is not available."""

import base64
import hashlib
import os

ENCRYPTION_KEY_ENV = "MEMORY_ENCRYPTION_KEY"


def _get_key() -> bytes | None:
    """Get 32-byte encryption key from environment."""
    raw = os.environ.get(ENCRYPTION_KEY_ENV)
    if not raw:
        return None
    # Accept hex-encoded 32-byte key or derive from passphrase
    try:
        key = bytes.fromhex(raw)
        if len(key) == 32:
            return key
    except ValueError:
        pass
    # Derive key from passphrase using SHA-256
    return hashlib.sha256(raw.encode()).digest()


def is_encryption_configured() -> bool:
    return _get_key() is not None


def encrypt(plaintext: str) -> bytes:
    """Encrypt text using AES-256-GCM. Returns nonce + ciphertext + tag."""
    key = _get_key()
    if key is None:
        raise RuntimeError(f"{ENCRYPTION_KEY_ENV} not set")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RuntimeError("cryptography package required for encryption: pip install cryptography")

    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return nonce + ciphertext  # 12 bytes nonce + ciphertext + 16 bytes tag


def decrypt(data: bytes) -> str:
    """Decrypt AES-256-GCM encrypted data."""
    key = _get_key()
    if key is None:
        raise RuntimeError(f"{ENCRYPTION_KEY_ENV} not set")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RuntimeError("cryptography package required for encryption: pip install cryptography")

    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode()


def encrypt_b64(plaintext: str) -> str:
    """Encrypt and return base64-encoded string."""
    return base64.b64encode(encrypt(plaintext)).decode()


def decrypt_b64(data: str) -> str:
    """Decrypt from base64-encoded string."""
    return decrypt(base64.b64decode(data))
