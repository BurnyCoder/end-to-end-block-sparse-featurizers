"""Global: full audit trails without secrets; local: test the reused sanitizer."""

from e2e_bsf.logging_utils import sanitize_for_logging


def test_long_outputs_preserved_but_secrets_redacted():
    """No length cap may truncate a future LLM output or hide a failed prediction."""
    text = "complete-output " * 10000
    sanitized = sanitize_for_logging({"output": text, "HF_TOKEN": "hf_abcdefghijklmn"})
    assert sanitized["output"] == text
    assert sanitized["HF_TOKEN"] == "[REDACTED]"
