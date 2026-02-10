#!/usr/bin/env bash
# Quantum Cracker -- Preflight Check
# Run at the start of every session

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }

echo "========================================="
echo " Quantum Cracker -- Preflight Check"
echo "========================================="
echo ""

# 1. Git status
echo "--- Git ---"
BRANCH=$(git branch --show-current 2>/dev/null || echo "none")
if [ "$BRANCH" != "none" ]; then
    pass "On branch: $BRANCH"
else
    fail "Not in a git repo"
fi

DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [ "$DIRTY" -eq 0 ]; then
    pass "Working tree clean"
else
    warn "$DIRTY uncommitted change(s)"
    git status --short
fi

LAST_COMMIT=$(git log --oneline -1 2>/dev/null || echo "no commits yet")
echo "  Last commit: $LAST_COMMIT"
echo ""

# 2. Python environment
echo "--- Python ---"
if [ -d ".venv" ]; then
    pass "Virtual environment exists (.venv/)"
    if [ -f ".venv/bin/python" ]; then
        PY_VER=$(.venv/bin/python --version 2>&1)
        pass "Python: $PY_VER"
    fi
else
    warn "No .venv/ found -- run: python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
fi
echo ""

# 3. Key files check
echo "--- Key Files ---"
for f in CLAUDE.md PROGRESS.md DECISIONS.md BLUEPRINTS.md REBUILD.md pyproject.toml; do
    if [ -f "$f" ]; then
        pass "$f"
    else
        fail "$f missing"
    fi
done
echo ""

# 4. Progress summary
echo "--- Progress ---"
if [ -f "PROGRESS.md" ]; then
    DONE=$(grep -c '\[x\]' PROGRESS.md 2>/dev/null || echo 0)
    WIP=$(grep -c '\[~\]' PROGRESS.md 2>/dev/null || echo 0)
    TODO=$(grep -c '\[ \]' PROGRESS.md 2>/dev/null || echo 0)
    echo "  Done: $DONE | In Progress: $WIP | Todo: $TODO"
fi
echo ""

echo "========================================="
echo " Preflight complete"
echo "========================================="
