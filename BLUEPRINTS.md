# Quantum Cracker - Blueprints

Copy-paste prompts for common tasks. Check here before writing from scratch.

---

## Add a new quantum experiment
```
Create a new experiment in src/quantum_cracker/core/ that:
1. Defines the quantum circuit
2. Prepares superposition states
3. Measures specified observables
4. Returns structured results dict
Follow the pattern in existing experiments. Add tests in tests/.
```

## Add a new analysis module
```
Create a new analysis module in src/quantum_cracker/analysis/ that:
1. Takes experiment results as input
2. Computes the relevant metrics
3. Returns a summary dict with statistics
4. Includes visualization helper if applicable
Add tests in tests/.
```

## Add a new visualization
```
Create a new visualization in src/quantum_cracker/visualization/ that:
1. Takes analysis results as input
2. Creates matplotlib figure(s)
3. Supports both display and save-to-file modes
4. Follows existing plot style conventions
```

## Run a full experiment pipeline
```
1. Run preflight: bash scripts/preflight.sh
2. Execute experiment: python -m quantum_cracker.core.<experiment>
3. Run analysis: python -m quantum_cracker.analysis.<analyzer>
4. Generate plots: python -m quantum_cracker.visualization.<plotter>
5. Export results: CSV to ~/Desktop
```
