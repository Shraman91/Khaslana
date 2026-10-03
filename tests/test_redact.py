"""Tests for core/redact.py — the secret redaction engine."""

from khaslana.core.redact import is_high_entropy, redact, shannon_entropy


class TestShannonEntropy:
    def test_empty_string(self) -> None:
        assert shannon_entropy("") == 0.0

    def test_single_char(self) -> None:
        assert shannon_entropy("aaaa") == 0.0

    def test_high_entropy_random_string(self) -> None:
        # A string of unique characters has high entropy
        s = "abcdefghijklmnopqrstuvwxyz0123456789"
        assert shannon_entropy(s) > 4.0

    def test_low_entropy_repetitive(self) -> None:
        s = "aaabbbccc"
        assert shannon_entropy(s) < 2.0


class TestHighEntropyDetection:
    def test_short_token_not_flagged(self) -> None:
        assert not is_high_entropy("hello")

    def test_long_random_hex_flagged(self) -> None:
        # 48-hex chars — random-looking
        token = "a3b9f1c8e7d04a6b2c5d8e9f0a1b3c4d5e6f7a8b9c0d1e2f"
        assert is_high_entropy(token)

    def test_dictionary_word_not_flagged(self) -> None:
        assert not is_high_entropy("helloworldthisisatest!!!!")

    def test_url_paths_not_flagged(self) -> None:
        # Contains slashes, so skip
        assert not is_high_entropy("https://example.com/very/long/path/to/resource/here")


class TestPatternRedaction:
    def test_openai_key(self) -> None:
        text = "My key is sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
        result = redact(text)
        assert "sk-proj-" not in result
        assert "[REDACTED:OPENAI_KEY]" in result

    def test_anthropic_key(self) -> None:
        text = "key=sk-ant-aBcDeFgHiJkLmNoPqRsTuVwXyZ012"
        result = redact(text)
        assert "sk-ant-" not in result

    def test_google_api_key(self) -> None:
        text = "Using key AIzaSyC0example1234567890abcdefghijkl"
        result = redact(text)
        assert "AIza" not in result
        assert "[REDACTED:GOOGLE_KEY]" in result

    def test_aws_access_key(self) -> None:
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_github_pat(self) -> None:
        text = "token = ghp_ABCDefgh12345678901234567890123456"
        result = redact(text)
        assert "ghp_" not in result

    def test_slack_token(self) -> None:
        text = "export SLACK=xoxb-123456789-abcdefghijklmnop"
        result = redact(text)
        assert "xoxb-" not in result

    def test_jwt_token(self) -> None:
        header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        payload = "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
        sig = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        jwt = f"{header}.{payload}.{sig}"
        text = f"Authorization: Bearer {jwt}"
        result = redact(text)
        assert header not in result

    def test_connection_uri_postgres(self) -> None:
        text = "DATABASE_URL=postgresql://admin:s3cretPass@db.example.com:5432/mydb"
        result = redact(text)
        assert "s3cretPass" not in result
        assert "[REDACTED:PASSWORD]" in result
        # Host information should still be present for debugging
        assert "db.example.com" in result

    def test_connection_uri_redis(self) -> None:
        text = "REDIS_URL=redis://default:mypassword@redis.cloud.com:6379/0"
        result = redact(text)
        assert "mypassword" not in result
        assert "[REDACTED:PASSWORD]" in result

    def test_key_value_pair(self) -> None:
        text = "api_key=super_S3cret_Key_1234567890"
        result = redact(text)
        assert "super_S3cret_Key_1234567890" not in result

    def test_key_value_trivial_not_redacted(self) -> None:
        text = "password=none"
        result = redact(text)
        assert result == text  # 'none' is a known placeholder

    def test_bearer_token(self) -> None:
        token = "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWRtaW4ifQ.dummysignaturevalue"
        text = f"Authorization: Bearer {token}"
        result = redact(text)
        assert "Bearer [REDACTED" in result

    def test_private_key_block(self) -> None:
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA0Z\n-----END RSA PRIVATE KEY-----"
        result = redact(text)
        assert "MIIEpAIBAAKCAQEA0Z" not in result
        assert "[REDACTED:PRIVATE_KEY]" in result


class TestEntropyRedaction:
    def test_random_token_redacted(self) -> None:
        token = "a3b9f1c8e7d04a6b2c5d8e9f0a1b3c4d5e6f7a8b9c0d1e2f"
        text = f"token is {token} found here"
        result = redact(text)
        assert token not in result

    def test_normal_text_not_redacted(self) -> None:
        text = "This is a normal paragraph about coding and software engineering."
        result = redact(text)
        assert result == text


class TestNoSecretPassthrough:
    def test_empty_input(self) -> None:
        assert redact("") == ""

    def test_passthrough_safe_text(self) -> None:
        safe = "def hello():\n    print('world')\n"
        assert redact(safe) == safe

    def test_multiple_secrets_in_one_text(self) -> None:
        text = (
            "OPENAI_KEY=sk-proj-ABC123DEF456GHI789JKL012MNO345P\n"
            "DB=postgres://user:password@host:5432/db\n"
            "GOOGLE=AIzaSyC0example1234567890abcdefghijkl\n"
        )
        result = redact(text)
        assert "sk-proj-" not in result
        assert "password" not in result
        assert "AIza" not in result
