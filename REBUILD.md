# Quantum Cracker - Rebuild Manual

Complete instructions to rebuild this project from scratch on a new machine.

---

## Prerequisites
- Python 3.12+
- git
- pip

## Steps

### 1. Clone the repo
```bash
git clone <repo-url> ~/quantum-cracker
cd ~/quantum-cracker
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e ".[dev]"
```

### 4. Verify installation
```bash
pytest
python -m quantum_cracker --version
```

### 5. Set up Claude Code memory
```bash
# Memory files are at ~/.claude/projects/-Users-kjm-quantum-cracker/memory/
# They should be created automatically on first Claude Code session
```

## Environment Variables
```bash
# None required yet -- add here as needed
```

## Data Files
```
data/   # Experiment configs and input data go here
```
