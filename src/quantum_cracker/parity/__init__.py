"""Parity-Driven Quantum Mechanics engine for ECDLP.

Maps the discrete logarithm problem to a Z2 Ising Hamiltonian
and solves via parity-weighted dynamics (pair hopping, coherence
asymmetry, anchoring).
"""

from __future__ import annotations

from quantum_cracker.parity.types import (
    AnnealResult,
    AnnealSchedule,
    DynamicsSnapshot,
    OracleResult,
    ParityConfig,
)

__all__ = [
    "AnnealResult",
    "AnnealSchedule",
    "DynamicsSnapshot",
    "OracleResult",
    "ParityConfig",
]
