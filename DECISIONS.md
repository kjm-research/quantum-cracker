# Quantum Cracker - Architecture Decision Log

Check here before re-debating a settled choice.

---

## ADR-001: Python as primary language
**Date:** 2026-02-09
**Status:** Accepted
**Context:** Need a language with strong quantum computing and scientific computing libraries.
**Decision:** Python 3.12+ with Qiskit, NumPy, SciPy, Matplotlib.
**Rationale:** De facto standard for quantum computing research. Qiskit provides IBM quantum hardware access. NumPy/SciPy for numerical analysis. Matplotlib for visualization.

## ADR-002: Project structure
**Date:** 2026-02-09
**Status:** Accepted
**Context:** Need organized layout for simulation code, analysis, and experiments.
**Decision:** Single-package `src/quantum_cracker/` layout with `core/`, `analysis/`, `visualization/`, `utils/` subpackages.
**Rationale:** Clean separation of concerns. `src/` layout prevents accidental imports of uninstalled code. Subpackages map to logical domains.
