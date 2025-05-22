"""Debug script for SysMetricsRunContextProvider.

This script manually starts an MLflow run and logs the system-level tags
captured by the sysmetrics plugin.

Usage:
    poetry run python scripts/debug_run.py

Note:
    - Set the environment variable: MLFLOW_RUN_CONTEXT_PROVIDER=sysmetrics
    - Intended for manual debugging and development only.

"""

import logging
import mlflow

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run a manual MLflow experiment to inspect sysmetrics tags."""
    mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run() as run:
        mlflow.log_param("debug_param", 42)
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run.info.run_id)

        logger.info("Captured system tags:\n" + "-" * 40)
        for key, value in run_data.data.tags.items():
            if key.startswith("sys."):
                logger.info(f"{key}: {value}")
        logger.info("-" * 40)


if __name__ == "__main__":
    main()
