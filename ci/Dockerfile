# Build agent for Jenkins (and the "agent image" job in GitHub Actions).
# Python 3.12 + a C++17 toolchain; CMake and pybind11 come from pip.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NETSENTINEL_LLM=none
