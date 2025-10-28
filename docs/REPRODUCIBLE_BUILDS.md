# Reproducible Builds Guide

## Overview

This document describes how to create reproducible builds of the ShivX AI Trading System. Reproducible builds ensure that the same source code always produces bit-for-bit identical binaries, allowing anyone to verify that distributed binaries match the source code.

## Why Reproducible Builds?

- **Security**: Verify that binaries haven't been tampered with
- **Trust**: Allow users to independently verify build authenticity
- **Debugging**: Ensure development and production builds match exactly
- **Compliance**: Meet security audit requirements

## Build Determinism Requirements

### 1. Environment Specification

To achieve reproducible builds, you must use the exact same build environment:

```yaml
Build Environment:
  OS: Ubuntu 22.04 LTS
  Python: 3.10.12
  pip: 23.3.1
  Docker: 24.0.7
  Docker Buildx: 0.12.0
  Node.js: 18.19.0 (if building frontend)
  npm: 10.2.3 (if building frontend)
```

### 2. Dependency Pinning

All dependencies are pinned to exact versions in `requirements.txt`. Never use version ranges (e.g., `>=`, `~=`) in production builds.

**Example from requirements.txt:**
```python
fastapi==0.109.0  # Good: Exact version
uvicorn[standard]==0.27.0  # Good: Exact version
# NOT: fastapi>=0.109.0  # Bad: Version range
```

### 3. Docker Build Configuration

Use the following Docker build arguments for reproducibility:

```dockerfile
# Base image with exact version
FROM python:3.10.12-slim-bookworm

# Set SOURCE_DATE_EPOCH for reproducibility
ARG SOURCE_DATE_EPOCH=1704067200
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

# Disable pip cache
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
```

## Step-by-Step Build Instructions

### Prerequisites

1. **Install exact Python version:**
   ```bash
   # Using pyenv (recommended)
   pyenv install 3.10.12
   pyenv local 3.10.12

   # Verify
   python --version  # Should output: Python 3.10.12
   ```

2. **Install exact pip version:**
   ```bash
   python -m pip install --upgrade pip==23.3.1
   ```

3. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/shivx.git
   cd shivx
   git checkout v2.0.0  # Use specific tag/commit
   ```

### Build Process

#### Option 1: Docker Build (Recommended)

```bash
# Set reproducible build timestamp
export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)

# Build Docker image
docker build \
  --build-arg SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
  --build-arg PYTHON_VERSION=3.10.12 \
  --build-arg BUILD_DATE=$(date -u -d @${SOURCE_DATE_EPOCH} +"%Y-%m-%dT%H:%M:%SZ") \
  -t shivx:reproducible \
  -f Dockerfile \
  .

# Save image
docker save shivx:reproducible | gzip > dist/shivx-docker-v2.0.0.tar.gz
```

#### Option 2: Python Package Build

```bash
# Create clean virtual environment
python -m venv venv
source venv/bin/activate

# Install exact dependencies
pip install --no-cache-dir -r requirements.txt

# Build Python package
python -m build --no-isolation

# Output: dist/shivx-2.0.0.tar.gz and dist/shivx-2.0.0-py3-none-any.whl
```

#### Option 3: Windows Executable Build

```bash
# On Linux with Wine
./scripts/build_windows.sh --clean

# On Windows
.\scripts\build_windows.ps1 -Clean

# Output: dist/ShivX.exe
```

## Hash Verification

After building, generate and verify hashes:

### Generate Hashes

```bash
# Generate SHA256 hashes
./scripts/generate_hashes.sh

# This creates:
# - release/artifacts/hashes.sha256
# - release/artifacts/hashes.sha512
```

### Verify Hashes

```bash
# Verify all artifacts
./scripts/verify_hashes.sh

# Expected output:
# ✓ dist/shivx-docker-v2.0.0.tar.gz: OK
# ✓ dist/shivx-2.0.0.tar.gz: OK
# ✓ dist/shivx-2.0.0-py3-none-any.whl: OK
# ✓ dist/ShivX.exe: OK
```

## Known Hash Values (v2.0.0)

These are the expected SHA256 hashes for version 2.0.0 builds:

```
# Docker image
sha256:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  shivx-docker-v2.0.0.tar.gz

# Python package (source)
sha256:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  shivx-2.0.0.tar.gz

# Python wheel
sha256:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  shivx-2.0.0-py3-none-any.whl

# Windows executable
sha256:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  ShivX.exe
```

## Troubleshooting Non-Reproducible Builds

If your build produces different hashes, check:

### 1. Python Version Mismatch

```bash
python --version
# Must be exactly: Python 3.10.12
```

### 2. Different Dependency Versions

```bash
pip freeze | diff - <(cat requirements.txt | grep -v "#")
# Should show no differences
```

### 3. System Timezone/Locale

```bash
# Set UTC timezone
export TZ=UTC
export LC_ALL=C
export LANG=C.UTF-8
```

### 4. Git Metadata

```bash
# Ensure clean working directory
git status
# Should show: nothing to commit, working tree clean

# Check current commit
git rev-parse HEAD
# Should match the commit you're building from
```

### 5. Build Timestamp

```bash
# Use SOURCE_DATE_EPOCH from git
export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
echo $SOURCE_DATE_EPOCH
```

## CI/CD Integration

The CI/CD pipeline automatically creates reproducible builds:

```yaml
# .github/workflows/release.yml
- name: Build with reproducibility
  env:
    SOURCE_DATE_EPOCH: ${{ steps.get-date.outputs.epoch }}
  run: |
    docker build \
      --build-arg SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
      -t shivx:${{ github.ref_name }} \
      .
```

## Verification Script

Use the automated verification script:

```bash
# Build and verify
./scripts/reproducible_build_verify.sh v2.0.0

# This will:
# 1. Build from clean state
# 2. Generate hashes
# 3. Build again
# 4. Compare hashes
# 5. Report if build is reproducible
```

## Advanced: Multi-Stage Docker Build

For maximum reproducibility, use multi-stage builds:

```dockerfile
# Stage 1: Build environment
FROM python:3.10.12-slim-bookworm AS builder
ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# Stage 2: Runtime
FROM python:3.10.12-slim-bookworm
ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

COPY --from=builder /build/deps /usr/local/lib/python3.10/site-packages
COPY . /app
WORKDIR /app

CMD ["python", "main.py"]
```

## Best Practices

1. **Always use git tags for releases**
   ```bash
   git tag -a v2.0.0 -m "Release version 2.0.0"
   git push origin v2.0.0
   ```

2. **Document build environment in release notes**
   - Python version
   - OS version
   - Docker version
   - Build timestamp

3. **Publish hashes with releases**
   - Include SHA256 hashes in GitHub releases
   - Sign hashes with GPG

4. **Test reproducibility regularly**
   ```bash
   # Automated test
   make reproducible-test
   ```

5. **Use Docker for consistent environments**
   - Eliminates system-specific variations
   - Provides identical build environment

## Resources

- [Reproducible Builds Project](https://reproducible-builds.org/)
- [SOURCE_DATE_EPOCH Specification](https://reproducible-builds.org/docs/source-date-epoch/)
- [Docker Build Reproducibility](https://docs.docker.com/build/building/best-practices/#reproducible-builds)

## Support

If you have questions about reproducible builds:
- Open an issue: https://github.com/yourusername/shivx/issues
- Documentation: https://docs.shivx.ai/reproducible-builds
- Contact: devops@shivx.ai

---

**Last Updated:** 2025-10-28
**Version:** 2.0.0
**Maintained by:** ShivX DevOps Team
