"""Unit tests for macOS GPU chipset detection utility in mlflow-sysmetrics.

This test suite validates the behavior of get_macos_gpu_chipset() to ensure
correct parsing and fallback logic when invoking system_profiler.

These tests rely on monkeypatching subprocess behavior and are safe to run
cross-platform.

"""

import subprocess
import pytest
from mlflow_sysmetrics.utils.mac import get_macos_gpu_chipset


@pytest.mark.unit
def test_get_macos_gpu_chipset_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the chipset model is correctly parsed from system_profiler output.

    This simulates macOS output containing a valid `Chipset Model` entry.

    Args:
        monkeypatch: Pytest fixture to override subprocess.check_output.

    Asserts:
        That the function returns the extracted GPU name from mock output.

    """
    mock_output = (
        "Graphics/Displays:\n"
        "    Apple M2 Pro:\n"
        "      Chipset Model: Apple M2 Pro\n"
        "      Type: GPU\n"
    )

    def mock_check_output(*args, **kwargs) -> str:
        return mock_output

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    result: str = get_macos_gpu_chipset()
    assert result == "Apple M2 Pro"


@pytest.mark.unit
def test_get_macos_gpu_chipset_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that 'None' is returned if system_profiler fails or returns malformed output.

    Args:
        monkeypatch: Pytest fixture to simulate subprocess errors.

    Asserts:
        That the function returns 'None' on subprocess or parsing failure.

    """
    def mock_check_output(*args, **kwargs) -> str:
        msg = "mock error"
        raise subprocess.SubprocessError(msg)

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    result: str = get_macos_gpu_chipset()
    assert result == "None"
