# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with RHEL AI midstream extensions.

## [0.2.0+rhaiv.1] - 2026-03-04

### Changed

**New versioning scheme**: This release adopts the RHEL AI midstream versioning pattern `X.Y.Z+rhaiv.N` (e.g., `0.2.0+rhaiv.1`).

- Automatic version management via setuptools-scm (derived from git tags)
- Self-contained multi-stage Containerfile (works in all build environments)
- Tags are now immutable - never deleted or moved
- Build counter increments for fixes, resets on upstream version bumps

### For Users

- **Installation unchanged**: `pip install llama-stack-provider-trustyai-garak` continues to work
- **Version pinning**: Use full version string (e.g., `==0.2.0+rhaiv.1`) for reproducible builds
- **Container builds**: No changes required

See the [Versioning](README.md#versioning) section in README for details.

---

## [0.2.0] - 2026-02-XX

### Features

- Garak integration for LLM vulnerability scanning
- Llama Stack evaluation provider implementation
- EvalHub Kubeflow Pipelines support
- Predefined benchmarks (OWASP LLM Top 10, AVID taxonomy, quick scans)
- Shield testing capabilities
- TBSA (Tier-Based Security Aggregate) scoring
- Multiple deployment modes (total remote, partial remote, total inline)
- HTML and JSON report generation
