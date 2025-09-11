# Compatibility Matrix

This document tracks the compatibility of `llama-stack-provider-trustyai-garak` with different versions of [llama-stack](https://github.com/llamastack/llama-stack) and its dependencies across releases.

## Version Compatibility Table

| Provider Version | Llama-Stack Version | Python Version | Key Dependencies | Status | Notes |
|------------------|---------------------|----------------|------------------|---------|-------|
| 0.1.3 | ==0.2.18 | >=3.12 | greenlet, httpx[http2], kfp, kfp-kubernetes, kfp-server-api, boto3, garak | Current | Latest stable release with thin dependencies and lazy kfp & s3 client init for remote mode |
| 0.1.2 | >=0.2.15 | >=3.12 | fastapi, opentelemetry-api, opentelemetry-exporter-otlp, aiosqlite, greenlet, uvicorn, ipykernel, httpx[http2], kfp, kfp-kubernetes, kfp-server-api, boto3, garak | | Release with both remote and inline implementation |
| 0.1.1 | >=0.2.15 | >=3.12 | fastapi, opentelemetry-api, opentelemetry-exporter-otlp, aiosqlite, greenlet, uvicorn, ipykernel, httpx[http2], garak |  | Initial stable release with inline implementation |

## Dependency Details

### Core Dependencies

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