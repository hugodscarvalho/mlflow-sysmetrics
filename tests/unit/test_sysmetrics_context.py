"""Unit tests for the SysMetricsRunContextProvider MLflow plugin.

This test suite validates that the provider correctly captures system-level
metadata and returns it as MLflow run tags.

Only the logic of the provider is tested — no MLflow tracking is involved.
"""

import pytest

from mlflow_sysmetrics.system_context import SysMetricsRunContextProvider
from mlflow_sysmetrics.utils.constants import (
    TAG_CPU,
    TAG_CPU_CORES,
    TAG_MEMORY_GB,
    TAG_DISK_FREE_GB,
    TAG_PLATFORM,
    TAG_GPU,
    TAG_ERROR,
)


@pytest.fixture
def context_provider() -> SysMetricsRunContextProvider:
    """Fixture to initialize the context provider.

    Returns:
        SysMetricsRunContextProvider: An instance of the sysmetrics plugin class.

    """
    return SysMetricsRunContextProvider()


@pytest.mark.unit
def test_provider_is_always_in_context(context_provider: SysMetricsRunContextProvider) -> None:
    """Test that the provider is always considered in context.

    Args:
        context_provider: The SysMetricsRunContextProvider instance.

    Asserts:
        The provider should always return True for `in_context()`.

    """
    assert context_provider.in_context() is True


@pytest.mark.unit
def test_tags_include_core_metrics(context_provider: SysMetricsRunContextProvider) -> None:
    """Test that all expected system tags are returned by the provider.

    Args:
        context_provider: The SysMetricsRunContextProvider instance.

    Asserts:
        All required tag keys are present in the output.

    """
    tags: dict[str, str] = context_provider.tags()

    required_keys = {
        TAG_CPU,
        TAG_CPU_CORES,
        TAG_MEMORY_GB,
        TAG_DISK_FREE_GB,
        TAG_PLATFORM,
        TAG_GPU,
    }

    missing = required_keys - tags.keys()
    assert not missing, f"Missing tags: {missing}"


@pytest.mark.unit
def test_tag_value_formats(context_provider: SysMetricsRunContextProvider) -> None:
    """Test that tag values are in expected types and formats.

    Args:
        context_provider: The SysMetricsRunContextProvider instance.

    Asserts:
        Each tag is correctly typed and has valid content.

    """
    tags: dict[str, str] = context_provider.tags()

    assert isinstance(tags[TAG_CPU], str)
    assert tags[TAG_CPU] != ""

    assert tags[TAG_CPU_CORES].isdigit()
    assert float(tags[TAG_MEMORY_GB]) > 0
    assert float(tags[TAG_DISK_FREE_GB]) > 0
    assert isinstance(tags[TAG_PLATFORM], str)
    assert isinstance(tags[TAG_GPU], str)


@pytest.mark.unit
def test_tags_fail_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    context_provider: SysMetricsRunContextProvider,
) -> None:
    """Test that errors during tag collection are captured as sysmetrics.error.

    This simulates a failure in disk usage metrics.

    Args:
        monkeypatch: Pytest monkeypatch fixture to override disk usage.
        context_provider: The SysMetricsRunContextProvider instance.

    Asserts:
        That a 'sysmetrics.error' tag is present and includes the simulated message.

    """
    def broken_disk_usage(_: str) -> None:
        msg = "Disk error"
        raise RuntimeError(msg)

    monkeypatch.setattr("shutil.disk_usage", broken_disk_usage)

    tags: dict[str, str] = context_provider.tags()
    assert TAG_ERROR in tags
    assert "Disk error" in tags[TAG_ERROR]


@pytest.mark.unit
def test_gpu_tag_from_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the GPU tag is populated from nvidia-smi on non-macOS."""
    from mlflow_sysmetrics import system_context

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *args, **kwargs: b"NVIDIA GeForce RTX 4090",
    )

    provider = system_context.SysMetricsRunContextProvider()
    tags = provider.tags()

    assert tags[TAG_GPU] == "NVIDIA GeForce RTX 4090"


@pytest.mark.unit
def test_gpu_tag_fallback_when_nvidia_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the GPU tag is set to 'None' if nvidia-smi fails."""
    from mlflow_sysmetrics import system_context

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))

    provider = system_context.SysMetricsRunContextProvider()
    tags = provider.tags()

    assert tags[TAG_GPU] == "None"


@pytest.mark.unit
def test_gpu_tag_from_macos_chipset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the GPU tag uses get_macos_gpu_chipset() on macOS."""
    from mlflow_sysmetrics import system_context

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("mlflow_sysmetrics.system_context.get_macos_gpu_chipset", lambda: "Apple M2 Max")

    provider = system_context.SysMetricsRunContextProvider()
    tags = provider.tags()

    assert tags[TAG_GPU] == "Apple M2 Max"


@pytest.mark.unit
def test_gpu_tag_from_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the GPU tag uses get_windows_gpu_name() on Windows."""
    from mlflow_sysmetrics import system_context

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("mlflow_sysmetrics.system_context.get_windows_gpu_name", lambda: "NVIDIA GeForce RTX 4060")

    provider = system_context.SysMetricsRunContextProvider()
    tags = provider.tags()

    assert tags[TAG_GPU] == "NVIDIA GeForce RTX 4060"
