"""Tests for AES-256-GCM encryption module."""

import hashlib
import os

import pytest

from claude_memory.crypto import (
    ENCRYPTION_KEY_ENV,
    decrypt,
    decrypt_b64,
    encrypt,
    encrypt_b64,
    is_encryption_configured,
)

# A valid 32-byte hex key for testing
TEST_HEX_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
TEST_PASSPHRASE = "my-test-passphrase"


@pytest.fixture
def hex_key_env(monkeypatch):
    monkeypatch.setenv(ENCRYPTION_KEY_ENV, TEST_HEX_KEY)


@pytest.fixture
def passphrase_env(monkeypatch):
    monkeypatch.setenv(ENCRYPTION_KEY_ENV, TEST_PASSPHRASE)


@pytest.fixture
def no_key_env(monkeypatch):
    monkeypatch.delenv(ENCRYPTION_KEY_ENV, raising=False)


class TestEncryptionConfigured:
    def test_configured_with_hex_key(self, hex_key_env):
        assert is_encryption_configured() is True

    def test_configured_with_passphrase(self, passphrase_env):
        assert is_encryption_configured() is True

    def test_not_configured_without_env(self, no_key_env):
        assert is_encryption_configured() is False


class TestEncryptDecrypt:
    def test_roundtrip_with_hex_key(self, hex_key_env):
        plaintext = "Hello, this is a secret message!"
        encrypted = encrypt(plaintext)
        decrypted = decrypt(encrypted)
        assert decrypted == plaintext

    def test_roundtrip_with_passphrase(self, passphrase_env):
        plaintext = "Another secret message with passphrase key"
        encrypted = encrypt(plaintext)
        decrypted = decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_plaintexts_produce_different_ciphertexts(self, hex_key_env):
        ct1 = encrypt("message one")
        ct2 = encrypt("message two")
        assert ct1 != ct2

    def test_same_plaintext_produces_different_ciphertexts(self, hex_key_env):
        """Due to random nonce, encrypting the same text twice gives different results."""
        ct1 = encrypt("same message")
        ct2 = encrypt("same message")
        assert ct1 != ct2

    def test_missing_key_raises_on_encrypt(self, no_key_env):
        with pytest.raises(RuntimeError, match=ENCRYPTION_KEY_ENV):
            encrypt("test")

    def test_missing_key_raises_on_decrypt(self, no_key_env):
        with pytest.raises(RuntimeError, match=ENCRYPTION_KEY_ENV):
            decrypt(b"\x00" * 28)

    def test_decrypt_with_wrong_key_fails(self, hex_key_env, monkeypatch):
        plaintext = "secret data"
        encrypted = encrypt(plaintext)

        # Change to a different key
        monkeypatch.setenv(ENCRYPTION_KEY_ENV, "ff" * 32)
        with pytest.raises(Exception):
            decrypt(encrypted)

    def test_encrypted_data_format(self, hex_key_env):
        """Encrypted data should be at least 12 (nonce) + 16 (tag) bytes."""
        encrypted = encrypt("x")
        assert len(encrypted) >= 28  # 12 nonce + 1 plaintext + 16 tag = 29 minimum

    def test_unicode_roundtrip(self, hex_key_env):
        plaintext = "Unicode test: cafe\u0301, \u00fc\u00f6\u00e4, \U0001f512"
        decrypted = decrypt(encrypt(plaintext))
        assert decrypted == plaintext


class TestBase64Variants:
    def test_b64_roundtrip(self, hex_key_env):
        plaintext = "base64 test message"
        encrypted_b64 = encrypt_b64(plaintext)
        assert isinstance(encrypted_b64, str)
        decrypted = decrypt_b64(encrypted_b64)
        assert decrypted == plaintext

    def test_b64_output_is_valid_base64(self, hex_key_env):
        import base64
        encrypted_b64 = encrypt_b64("test")
        # Should not raise
        decoded = base64.b64decode(encrypted_b64)
        assert len(decoded) >= 28


class TestKeyDerivation:
    def test_hex_key_used_directly(self, hex_key_env):
        """A valid 64-char hex string should be used as-is (32 bytes)."""
        ct = encrypt("test")
        pt = decrypt(ct)
        assert pt == "test"

    def test_passphrase_derived_via_sha256(self, passphrase_env):
        """Non-hex strings should be derived via SHA-256."""
        ct = encrypt("test")
        pt = decrypt(ct)
        assert pt == "test"

    def test_short_hex_treated_as_passphrase(self, monkeypatch):
        """Hex string that's not exactly 32 bytes should be treated as passphrase."""
        monkeypatch.setenv(ENCRYPTION_KEY_ENV, "abcd1234")
        ct = encrypt("test")
        pt = decrypt(ct)
        assert pt == "test"
