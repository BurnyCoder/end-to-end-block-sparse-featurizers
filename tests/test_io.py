"""Global: recoverable progress. Local: reproduce transient Windows rename failures."""

from pathlib import Path
from e2e_bsf.io import atomic_replace


def test_atomic_replace_retries_windows_lock(monkeypatch, tmp_path):
    """A transient scanner lock must not abort a completed activation batch."""
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_text("complete")
    original = Path.replace
    attempts = []

    def locked(path, destination):
        """Simulate two Windows locks followed by a real atomic replacement."""
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError("transient sharing lock")
            error.winerror = 5
            raise error
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("e2e_bsf.io.time.sleep", lambda _: None)
    atomic_replace(source, target)
    assert target.read_text() == "complete" and len(attempts) == 3
