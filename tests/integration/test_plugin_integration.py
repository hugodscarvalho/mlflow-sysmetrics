"""Integration test for MLflow sysmetrics plugin.

This test verifies that when the plugin is activated via the
`MLFLOW_RUN_CONTEXT_PROVIDER` environment variable, MLflow correctly
logs system-level tags during a run.

Requires an accessible MLflow tracking server.
"""

import mlflow
import pytest

from mlflow_sysmetrics.utils.constants import TAG_CPU, TAG_MEMORY_GB


@pytest.mark.integration
def test_plugin_sysmetrics_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that sysmetrics plugin correctly logs system tags in an MLflow run.

    This test starts a run with the sysmetrics plugin enabled and asserts that
    key tags (e.g., CPU and memory info) are recorded in the run metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch): Utility to temporarily set environment variables.

    Raises:
        AssertionError: If expected tags are missing or have invalid values.

    """
    monkeypatch.setenv("MLFLOW_RUN_CONTEXT_PROVIDER", "sysmetrics")

    with mlflow.start_run() as run:
        client = mlflow.tracking.MlflowClient()
        tags: dict[str, str] = client.get_run(run.info.run_id).data.tags

    # Extract sys.* tags only
    sys_tags: dict[str, str] = {k: v for k, v in tags.items() if k.startswith("sys.")}

    assert TAG_CPU in sys_tags, f"Expected tag '{TAG_CPU}' missing"
    assert TAG_MEMORY_GB in sys_tags, f"Expected tag '{TAG_MEMORY_GB}' missing"
    assert sys_tags[TAG_CPU] != "", "CPU tag is unexpectedly empty"
    assert float(sys_tags[TAG_MEMORY_GB]) > 0, "Memory tag must be a positive number"
