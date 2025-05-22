"""Unit tests for Windows GPU name detection utility."""

import subprocess
import pytest
from mlflow_sysmetrics.utils.windows import get_windows_gpu_name


@pytest.mark.unit
def test_get_windows_gpu_name_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a valid GPU name is returned from mock PowerShell output."""
    mock_output = "NVIDIA GeForce RTX 4060\n"

    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: mock_output)

    result: str = get_windows_gpu_name()
    assert result == "NVIDIA GeForce RTX 4060"


@pytest.mark.unit
def test_get_windows_gpu_name_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that 'None' is returned when subprocess raises an error."""

    def raise_error(*a, **k):
        msg = "mock error"
        raise subprocess.SubprocessError(msg)

    monkeypatch.setattr("subprocess.check_output", raise_error)

    result: str = get_windows_gpu_name()
    assert result == "None"
