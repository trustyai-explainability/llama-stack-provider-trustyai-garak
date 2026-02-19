# Compatibility Matrix

This document tracks the compatibility of `llama-stack-provider-trustyai-garak` with different versions of [llama-stack](https://github.com/llamastack/llama-stack) and its dependencies across releases.

## Version Compatibility Table

| Provider Version | Llama-Stack Version | Python Version | Key Dependencies | Status | Notes |
|------------------|---------------------|----------------|------------------|---------|-------|
| 0.2.0 | >=0.5.0 | >=3.12 | kfp>=2.14.6, kfp-kubernetes>=2.14.6, kfp-server-api>=2.14.6, boto3>=1.35.88 | Current | Current release with updated metadata schema (`metadata.garak_config`) and remote/inline support |
| 0.1.3 | ==0.2.18 | >=3.12 | greenlet, httpx[http2], kfp, kfp-kubernetes, kfp-server-api, boto3, garak | | Latest stable release with thin dependencies and lazy kfp & s3 client init for remote mode |
| 0.1.2 | >=0.2.15 | >=3.12 | fastapi, opentelemetry-api, opentelemetry-exporter-otlp, aiosqlite, greenlet, uvicorn, ipykernel, httpx[http2], kfp, kfp-kubernetes, kfp-server-api, boto3, garak | | Release with both remote and inline implementation |
| 0.1.1 | >=0.2.15 | >=3.12 | fastapi, opentelemetry-api, opentelemetry-exporter-otlp, aiosqlite, greenlet, uvicorn, ipykernel, httpx[http2], garak |  | Initial stable release with inline implementation |

## Dependency Details

### Core Dependencies

#### Version 0.2.0 (latest)
- **llama-stack-client**: >=0.5.0
- **llama-stack-api**: >=0.5.0
- **llama-stack** (server extra): >=0.5.0
- **garak** (inline extra): ==0.14.0
- **kfp**: >=2.14.6
- **kfp-kubernetes**: >=2.14.6
- **kfp-server-api**: >=2.14.6
- **boto3**: >=1.35.88

#### Version 0.1.3
- **llama-stack**: == 0.2.18
- **greenlet**: Latest compatible (3.2.4)
- **httpx[http2]**: Latest compatible (0.28.1)
- **garak**: == 0.12.0
- **kfp**: Latest compatible (2.14.3)
- **kfp-kubernetes**: Latest compatible (2.14.3)
- **kfp-server-api**: Latest compatible (2.14.3)
- **boto3**: Latest compatible (1.40.27)

#### Version 0.1.2
- **llama-stack**: >=0.2.15
- **fastapi**: Latest compatible (0.116.1)
- **opentelemetry-api**: Latest compatible (1.36.0)
- **opentelemetry-exporter-otlp**: Latest compatible (1.36.0)
- **aiosqlite**: Latest compatible (0.21.0)
- **uvicorn**: Latest compatible (0.35.0)
- **ipykernel**: Latest compatible (6.30.1)
- **greenlet**: Latest compatible (3.2.4)
- **httpx[http2]**: Latest compatible (0.28.1)
- **garak**: == 0.12.0
- **kfp**: Latest compatible (2.14.3)
- **kfp-kubernetes**: Latest compatible (2.14.3)
- **kfp-server-api**: Latest compatible (2.14.3)
- **boto3**: Latest compatible (1.40.26)

#### Version 0.1.1
- **llama-stack**: >=0.2.15
- **fastapi**: Latest compatible (0.116.1)
- **opentelemetry-api**: Latest compatible (1.36.0)
- **opentelemetry-exporter-otlp**: Latest compatible (1.36.0)
- **aiosqlite**: Latest compatible (0.21.0)
- **greenlet**: Latest compatible (3.2.4)
- **uvicorn**: Latest compatible (0.35.0)
- **ipykernel**: Latest compatible (6.30.1)
- **httpx[http2]**: Latest compatible (0.28.1)
- **garak**: == 0.12.0

### Development Dependencies

All versions include the same development dependencies:
- **pytest**: Testing framework
- **pytest-asyncio**: Asynchronous tests
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **isort**: Import sorting

## Container Compatibility

The provider is built and compatible with:
- **Base Image**: `registry.access.redhat.com/ubi9/python-312:latest`
- **Llama-Stack Version**: 0.2.18 (in container builds)
- **Additional Runtime Dependencies**: torch, transformers, sqlalchemy, and others as specified in the Containerfile

## Image Compatibility (Latest Deployments)

Use the table below as a quick reference for image fields used in current remote deployments.

| Use Case | Config Key / Field | Where to Set | Recommended Image | Alternative | Notes |
|---|---|---|---|---|---|
| LLS distro image (total remote) | `spec.distribution.image` | `lsd_remote/llama_stack_distro-setup/lsd-garak.yaml` | `quay.io/opendatahub/llama-stack@sha256:cf21d3919d265f8796ed600bfe3d2eb3ce797b35ab8e60ca9b6867e0516675e5` | `quay.io/rhoai/odh-llama-stack-core-rhel9:rhoai-3.4` | Pick image matching your RHOAI/ODH release stream |
| Garak KFP base image (total remote) | `KUBEFLOW_GARAK_BASE_IMAGE` | `lsd_remote/llama_stack_distro-setup/lsd-config.yaml` | `quay.io/opendatahub/odh-trustyai-garak-lls-provider-dsp:dev` | `quay.io/rhoai/odh-trustyai-garak-lls-provider-dsp-rhel9:rhoai-3.4` | Injected into LSD env via `lsd-garak.yaml` |
| Garak KFP base image (partial remote) | `kubeflow_config.garak_base_image` (env: `KUBEFLOW_GARAK_BASE_IMAGE`) | `demos/2-partial-remote/partial-remote.yaml` | `quay.io/opendatahub/odh-trustyai-garak-lls-provider-dsp:dev` | `quay.io/rhoai/odh-trustyai-garak-lls-provider-dsp-rhel9:rhoai-3.4` | Used by KFP components for scan/parse/validate steps |

## Breaking Changes

### Version 0.1.3
- No breaking changes from 0.1.2
- Reduced dependencies and lazy client inits

### Version 0.1.2
- No breaking changes from 0.1.1
- Both remote and inline implementations are available

### Version 0.1.1
- Initial release with only inline implementation
- No breaking changes

## Future Planning

This compatibility matrix will be updated with each new release to include:
- New llama-stack version compatibility
- Dependency updates and changes
- Breaking changes and migration notes
- Container compatibility updates
- Testing status and known issues