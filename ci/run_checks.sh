#!/usr/bin/env bash
# One entry point for every CI system (GitHub Actions, Jenkins) and for local use.
#
#   ci/run_checks.sh setup   install Python dependencies
#   ci/run_checks.sh lint    ruff
#   ci/run_checks.sh cpp     build the C++ library and module, run GoogleTest
#   ci/run_checks.sh test    pytest (C++ parity tests too, if the module is built)
#   ci/run_checks.sh train   train, evaluate on held-out days, keep metrics.json
#   ci/run_checks.sh all     lint, cpp, test, train
#
# Test results (JUnit XML) and metrics go to $REPORTS_DIR (default reports/ci).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
REPORTS="${REPORTS_DIR:-reports/ci}"
mkdir -p "$REPORTS"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"

step() { printf '\n==> %s\n' "$*"; }

case "${1:-all}" in
  setup)
    step "Installing Python dependencies"
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements-dev.txt
    ;;
  lint)
    step "Lint (ruff)"
    "$PY" -m ruff check .
    ;;
  cpp)
    step "C++ build (warnings are errors)"
    cmake -S cpp -B build -DNETSENTINEL_WERROR=ON \
      -DPython_EXECUTABLE="$(command -v "$PY")" \
      -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
    cmake --build build -j "$JOBS"
    step "C++ unit tests (GoogleTest)"
    ctest --test-dir build --output-on-failure --output-junit "$PWD/$REPORTS/ctest.xml"
    ;;
  test)
    step "Python tests"
    "$PY" -m pytest -v --junitxml="$REPORTS/pytest.xml"
    ;;
  train)
    step "Train and evaluate on held-out days"
    rm -rf data models
    "$PY" model_trainer.py
    cp models/metrics.json "$REPORTS/metrics.json"
    ;;
  all)
    for s in lint cpp test train; do "$0" "$s"; done
    ;;
  *)
    echo "usage: $0 {setup|lint|cpp|test|train|all}" >&2
    exit 2
    ;;
esac
