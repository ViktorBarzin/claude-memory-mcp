"""Tests for credential detection and redaction."""


from claude_memory.credential_detector import (
    detect_credentials,
    is_sensitive,
    redact_credentials,
)


class TestDetectCredentials:
    def test_detect_postgres_connection_string(self):
        text = "db_url = postgres://user:pass@localhost:5432/mydb"
        creds = detect_credentials(text)
        assert len(creds) == 1
        assert creds[0].type == "connection_string"
        assert creds[0].confidence == 0.9
        assert "postgres://" in creds[0].matched_text

    def test_detect_password_assignment(self):
        text = 'password = "my_super_secret_pw"'
        creds = detect_credentials(text)
        assert len(creds) >= 1
        types = [c.type for c in creds]
        assert "password" in types

    def test_detect_api_key(self):
        text = "api_key = ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        creds = detect_credentials(text)
        assert len(creds) >= 1
        types = [c.type for c in creds]
        assert "api_key" in types

    def test_detect_private_key(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWep4PAtGoSo\n-----END RSA PRIVATE KEY-----"
        creds = detect_credentials(text)
        assert len(creds) == 1
        assert creds[0].type == "private_key"
        assert creds[0].confidence == 0.95

    def test_detect_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkw"
        creds = detect_credentials(text)
        assert len(creds) >= 1
        types = [c.type for c in creds]
        assert "bearer_token" in types

    def test_detect_aws_key(self):
        text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
        creds = detect_credentials(text)
        assert len(creds) >= 1
        types = [c.type for c in creds]
        assert "aws_key" in types

    def test_detect_github_token(self):
        text = "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        creds = detect_credentials(text)
        assert len(creds) >= 1
        types = [c.type for c in creds]
        assert "github_token" in types

    def test_no_false_positives_on_normal_text(self):
        text = "This is a normal paragraph about programming. It discusses variables, functions, and classes."
        creds = detect_credentials(text)
        assert len(creds) == 0

    def test_no_false_positives_on_short_password(self):
        # password values shorter than 8 chars should not match
        text = 'password = "short"'
        creds = detect_credentials(text)
        assert len(creds) == 0

    def test_min_confidence_filtering(self):
        text = 'secret = "abcdefghijklmnopqrstuvwxyz"'
        all_creds = detect_credentials(text, min_confidence=0.5)
        high_creds = detect_credentials(text, min_confidence=0.9)
        assert len(all_creds) >= len(high_creds)

    def test_overlapping_matches_keep_highest_confidence(self):
        # A text that could match both token and generic_secret
        text = 'secret = "abcdefghijklmnopqrstuvwxyz1234567890"'
        creds = detect_credentials(text, min_confidence=0.5)
        # Should not have overlapping ranges for the same span
        for i, c1 in enumerate(creds):
            for c2 in creds[i + 1:]:
                # No credential should be fully contained within another
                assert not (c1.start <= c2.start and c1.end >= c2.end)


class TestRedactCredentials:
    def test_redaction_replaces_with_marker(self):
        text = "db_url = postgres://user:pass@localhost:5432/mydb"
        creds = detect_credentials(text)
        redacted = redact_credentials(text, creds)
        assert "[REDACTED:connection_string]" in redacted
        assert "postgres://" not in redacted

    def test_redaction_preserves_surrounding_text(self):
        text = "before postgres://user:pass@localhost/db after"
        creds = detect_credentials(text)
        redacted = redact_credentials(text, creds)
        assert redacted.startswith("before ")
        assert redacted.endswith(" after")

    def test_redaction_no_credentials(self):
        text = "nothing sensitive here"
        redacted = redact_credentials(text, [])
        assert redacted == text

    def test_redaction_multiple_credentials(self):
        text = 'password = "mysecretpw123" and api_key = ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
        creds = detect_credentials(text)
        redacted = redact_credentials(text, creds)
        assert "mysecretpw123" not in redacted
        assert "[REDACTED:" in redacted


class TestIsSensitive:
    def test_sensitive_text(self):
        assert is_sensitive("password = supersecretvalue123")

    def test_non_sensitive_text(self):
        assert not is_sensitive("just a normal log message")

    def test_respects_min_confidence(self):
        text = 'secret = "abcdefghijklmnopqrstuvwxyz"'
        # Low confidence should detect
        assert is_sensitive(text, min_confidence=0.5)
        # Very high confidence should not detect generic_secret
        assert not is_sensitive(text, min_confidence=0.95)
