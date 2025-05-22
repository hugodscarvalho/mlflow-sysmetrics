# CHANGELOG

All notable changes to the `mlflow-sysmetrics` package will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and follows changelog conventions inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.1.3] – 2025-05-22
### Added
- Cross-platform GPU detection support:
  - **macOS**: Uses `system_profiler SPDisplaysDataType` to extract the GPU chipset model.
  - **Windows**: Uses PowerShell's `Get-CimInstance Win32_VideoController` to retrieve GPU name.
  - **Linux**: Continues to use `nvidia-smi` when available; otherwise falls back to `"None"`.

### Changed
- Refactored GPU detection logic into OS-specific utility modules for better maintainability.
- Added constants for GPU detection and subprocess timeouts to `constants.py`.
- Improved `README.md` to clarify GPU detection behavior on each platform.

### Notes
- This enhancement improves observability across all major platforms without introducing new dependencies.

## [v0.1.2] – 2025-05-22
### Changed
- Bumped patch version to trigger v0.1.2 deployment to PyPI.

## [v0.1.1] – 2025-05-22
### Changed
- Bumped patch version to trigger deployment to PyPI with no functional code changes.

## [v0.1.0] – 2025-05-22
### Added
- Initial release of the `mlflow-sysmetrics` plugin for MLflow.
- Implemented a custom `RunContextProvider` that logs system-level metrics as MLflow run tags.
- Tags include CPU model, core count, total memory, available disk, OS platform, and GPU (via `nvidia-smi`).
- Support for graceful failure and fallback for unsupported environments.
- Unit and integration test suites using `pytest` and `pytest-cov`.
- GitHub Actions CI/CD pipeline for linting, testing, and publishing to PyPI.

### Notes
- This version is a fully working, cross-platform, plugin-compatible MLflow extension designed to enrich run metadata for reproducibility and observability.
- Requires Python 3.9+, MLflow 2+, and `psutil`.