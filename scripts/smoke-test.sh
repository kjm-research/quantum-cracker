#!/usr/bin/env bash
# Quantum Cracker -- Smoke Test
# Run after making changes to verify nothing is broken

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; ERRORS=$((ERRORS+1)); }

ERRORS=0

echo "========================================="
echo " Quantum Cracker -- Smoke Test"
echo "========================================="
echo ""

# 1. Key files exist
echo "--- Key Files ---"
for f in CLAUDE.md PROGRESS.md DECISIONS.md pyproject.toml src/quantum_cracker/__init__.py; do
    if [ -f "$f" ]; then
        pass "$f"
    else
        fail "$f missing"
    fi
done
echo ""

# 2. Python import check
echo "--- Import Check ---"
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    if .venv/bin/python -c "import quantum_cracker; print(f'v{quantum_cracker.__version__}')" 2>/dev/null; then
        pass "quantum_cracker imports OK"
    else
        fail "quantum_cracker import failed"
    fi
else
    warn "No .venv -- skipping import check"
fi
echo ""

# 3. Tests
echo "--- Tests ---"
if [ -d ".venv" ] && [ -f ".venv/bin/pytest" ]; then
    if .venv/bin/pytest --tb=short -q 2>&1; then
        pass "Tests passed"
    else
        fail "Tests failed"
    fi
else
    warn "No .venv or pytest -- skipping tests"
fi
echo ""

# 4. Lint check
echo "--- Lint ---"
if [ -d ".venv" ] && [ -f ".venv/bin/ruff" ]; then
    if .venv/bin/ruff check src/ 2>&1; then
        pass "Ruff lint clean"
    else
        warn "Ruff lint issues found"
    fi
else
    warn "No ruff -- skipping lint"
fi
echo ""

echo "========================================="
if [ "$ERRORS" -eq 0 ]; then
    echo -e " ${GREEN}Smoke test passed${NC}"
else
    echo -e " ${RED}$ERRORS error(s) found${NC}"
fi
echo "========================================="
exit $ERRORS
