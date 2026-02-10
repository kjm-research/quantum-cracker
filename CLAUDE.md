# Quantum Cracker - Claude Code Reference

## Project Overview
Software to prove quantum superposition and its relational imprint on observable metrics.

## Project Location
- **Path:** /Users/kjm/quantum-cracker
- **Type:** Python project (NumPy, SciPy, Qiskit)

## Tech Stack
- **Language:** Python 3.12+
- **Quantum:** Qiskit for quantum circuit simulation
- **Scientific:** NumPy, SciPy, Matplotlib
- **Testing:** pytest
- **Package Management:** pip + pyproject.toml

## Project Structure
```
quantum-cracker/
  src/
    quantum_cracker/
      __init__.py
      core/           # Core quantum simulation logic
      analysis/       # Observable metric analysis
      visualization/  # Plotting and visualization
      utils/          # Shared utilities
  tests/              # Test suite
  data/               # Input data and experiment configs
  notebooks/          # Jupyter notebooks for exploration
  scripts/            # CLI scripts (preflight, smoke-test, etc.)
  docs/               # Documentation
```

## Commands
- **Run tests:** `pytest`
- **Run a script:** `python -m quantum_cracker.<module>`
- **Install deps:** `pip install -e ".[dev]"`

## Development Workflow
1. **Start session:** Run `bash scripts/preflight.sh`
2. **Starting a feature:** Update "Current Session" in memory MEMORY.md
3. **At breakpoints:** WIP commits (e.g. `[WIP] Add tomography analysis`)
4. **After changes:** Run `bash scripts/smoke-test.sh`
5. **On finishing:** Update PROGRESS.md + REBUILD.md + DECISIONS.md, clean commit, clear Current Session
6. **Session dies:** Next session reads memory + `git status` + `git diff` to recover

## User Preferences
- No emojis in code or output
- CSV reports to ~/Desktop
- Commit and push after features
- Terminal/CLI workflows preferred
- Co-author: `Co-Authored-By: Claude <noreply@anthropic.com>`

## Key Decisions
- See DECISIONS.md for architecture decision log
- Check BLUEPRINTS.md before writing common patterns from scratch
