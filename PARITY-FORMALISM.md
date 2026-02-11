# Parity-Driven Quantum Mechanics: Mathematical Formalism

**KJ M**

February 2026

**Abstract.** This document provides the complete mathematical formalism for the parity-driven quantum mechanics framework presented in the companion paper (*Parity-Driven Quantum Mechanics: A Framework for Superposition, Collapse, and the Gravity-Quantum Divide*, KJ M, 2026). We define the full field content, construct the Lagrangian, derive the equations of motion, present exact solutions for small particle-number systems, and establish the consistency of the theory with Lorentz invariance, energy conservation, the weak equivalence principle, and the recovery of standard quantum mechanics in appropriate limits. Two free parameters---the parity energy gap $\Delta_E$ and the anchoring coupling $\kappa$---determine all testable predictions.

**Keywords:** quantum foundations, parity formalism, Z2 symmetry, many-interacting-worlds, modified gravity, Lagrangian field theory, multiverse hopping

---

## 1. Notation and Conventions

Throughout this document we adopt the following conventions:

- **Natural units:** $\hbar = c = 1$ unless explicitly restored for physical estimates.
- **Spacetime indices:** Greek letters $\mu, \nu, \rho, \sigma = 0, 1, 2, 3$ label spacetime coordinates.
- **Universe labels:** Latin letters $\alpha, \beta, \gamma = 1, 2, \ldots, K$ label distinct universes.
- **Metric signature:** $(-,+,+,+)$, so that $\eta_{\mu\nu} = \text{diag}(-1,+1,+1,+1)$.
- **Einstein summation convention:** Repeated spacetime indices are summed unless stated otherwise. Universe-label sums are always written explicitly.
- **Gamma matrices:** Satisfy the Clifford algebra $\{\gamma^\mu, \gamma^\nu\} = 2\eta^{\mu\nu}$.
- **Charge conjugation:** $\psi_c = C\bar{\psi}^T$ with $C = i\gamma^2\gamma^0$ in the Dirac representation.
- **Hermitian conjugate:** Denoted $\dagger$. Complex conjugate: $*$. Dirac adjoint: $\bar{\psi} = \psi^\dagger \gamma^0$.
- **Notation for sums over graph neighbors:** $\langle\alpha,\beta\rangle$ denotes a sum over edges of the multiverse graph $G_K$, counting each edge once. $\beta \sim \alpha$ denotes a sum over all universes $\beta$ adjacent to $\alpha$.
- **Coordination number:** $z$ denotes the number of nearest neighbors of a vertex on $G_K$ (assumed uniform for a regular graph).

---

## 2. Field Content

### 2.1 Universe Label Space

The multiverse is modeled as a discrete graph $G_K = (V, E)$ with:

- **Vertices:** $V = \{1, 2, \ldots, K\}$, each representing a distinct universe.
- **Edges:** $E \subseteq V \times V$, each representing a transit channel between two universes.

For a regular lattice in $d$ dimensions with lattice spacing $a$, the coordination number is $z = 2d$. The continuum limit $K \to \infty$, $a \to 0$ with $t_1 a^2 = \hbar^2 / (2m)$ held fixed recovers the Schrodinger equation (Section 7.5).

The graph Laplacian is:

$$(\Delta_G f)_\alpha = \sum_{\beta \sim \alpha} (f_\beta - f_\alpha) = \sum_{\beta \sim \alpha} f_\beta - z \, f_\alpha$$

### 2.2 Fermion Field

In each universe $\alpha$, a Dirac spinor field $\psi_\alpha(x,t)$ represents the fundamental fermionic matter. The total fermion number operator in universe $\alpha$ is:

$$N_\alpha = \int d^3x \; \psi_\alpha^\dagger(x,t) \, \psi_\alpha(x,t)$$

This counts all fermions (particles and antiparticles, all species) present in universe $\alpha$ at time $t$. In the second-quantized theory, $N_\alpha$ has integer eigenvalues $n_\alpha \in \{0, 1, 2, \ldots\}$.

The total multiverse fermion number is conserved:

$$N_{\text{tot}} = \sum_{\alpha=1}^{K} N_\alpha = \text{const}$$

### 2.3 Parity Field (Z2 Order Parameter)

The parity of universe $\alpha$ is defined as:

$$\sigma_\alpha = (-1)^{N_\alpha} \in \{-1, +1\}$$

This is a **derived** quantity, not an independent degree of freedom. It is fully determined by $N_\alpha$.

- $\sigma_\alpha = +1$: Even parity. All particles paired. Classical-dominant regime.
- $\sigma_\alpha = -1$: Odd parity. At least one unpaired particle. Quantum-active regime.

This definition is consistent with the fermion parity superselection rule of Wick, Wightman, and Wigner (1952), which forbids coherent superpositions of states with different fermion parity: the superselection sectors are exactly the eigenspaces of $(-1)^{N_\alpha}$.

The total multiverse parity is:

$$\sigma_{\text{tot}} = \prod_{\alpha=1}^{K} \sigma_\alpha = (-1)^{N_{\text{tot}}}$$

This is a global invariant. If $N_{\text{tot}}$ is odd, at least one universe must always have odd parity.

### 2.4 Composite Boson (Pair) Field

We introduce a composite scalar field $\Phi_\alpha(x,t)$ representing a Cooper-pair-like condensate of paired fermions in universe $\alpha$. This field is a bound-state order parameter analogous to the BCS gap parameter.

In the mean-field approximation:

$$\langle \Phi_\alpha \rangle = \Delta_{\text{pair}} \, e^{i\theta_\alpha}$$

where $\Delta_{\text{pair}}$ is the pairing gap amplitude and $\theta_\alpha$ is the phase in universe $\alpha$.

$\Phi_\alpha$ is a complex scalar field with $U(1)$ phase symmetry. As a composite of two fermions, it obeys Bose statistics and can form macroscopic condensates. The pair number in universe $\alpha$ is:

$$N_\alpha^{\text{pair}} = \int d^3x \; |\Phi_\alpha(x,t)|^2$$

The number of unpaired fermions is $N_\alpha^{\text{unpaired}} = N_\alpha - 2 N_\alpha^{\text{pair}}$.

### 2.5 Metric Tensor

Each universe carries its own spacetime metric $g_{\mu\nu}^{(\alpha)}(x)$. In the weak-field limit:

$$g_{\mu\nu}^{(\alpha)} = \eta_{\mu\nu} + h_{\mu\nu}^{(\alpha)}$$

where $|h_{\mu\nu}^{(\alpha)}| \ll 1$. The key departure from general relativity is that the source of $h_{\mu\nu}^{(\alpha)}$ is restricted to paired matter only (Section 3.5).

### 2.6 Anchoring Field

The anchoring field $A_\alpha(x)$ quantifies the density of paired (classical) matter in the neighborhood of position $x$ in universe $\alpha$:

$$A_\alpha(x) = \sum_{I \in \text{macro}} f(|x - x_I|) \, n_I^{\text{paired}}$$

where the sum runs over macroscopic objects $I$ at positions $x_I$, $n_I^{\text{paired}}$ is the number of paired particles in object $I$, and $f(r)$ is a radial profile function normalized so that $\int f(r) \, d^3x = 1$.

More precisely, in the continuum limit:

$$A_\alpha(x) = \kappa_A \int d^3x' \; K(|x - x'|) \, n_{\text{pair}}(x')$$

where $K(r)$ is a kernel function (e.g., Gaussian or Yukawa profile with range $r_{\text{anchor}}$) and $n_{\text{pair}}(x') = |\Phi_\alpha(x')|^2$ is the local pair density. The anchoring field is not dynamical---it is slaved to the pair condensate.

---

## 3. The Lagrangian

The total Lagrangian of the multiverse is:

$$\boxed{\mathcal{L}_{\text{total}} = \sum_{\alpha=1}^{K} \left[\mathcal{L}_{\text{free}}^{(\alpha)} + \mathcal{L}_{\text{pair}}^{(\alpha)} + \mathcal{L}_{\text{parity}}^{(\alpha)} + \mathcal{L}_{\text{grav}}^{(\alpha)} + \mathcal{L}_{\text{anchor}}^{(\alpha)}\right] + \mathcal{L}_{\text{transit}}}$$

We now define each term.

### 3.1 Free Fermion Lagrangian

$$\mathcal{L}_{\text{free}}^{(\alpha)} = \bar{\psi}_\alpha \left(i \gamma^\mu D_\mu - m\right) \psi_\alpha$$

This is the standard Dirac Lagrangian for fermions in universe $\alpha$. The covariant derivative $D_\mu$ includes gauge field couplings (electromagnetic, weak, strong) within universe $\alpha$. Each universe is independently Lorentz covariant.

In the presence of a curved background:

$$\mathcal{L}_{\text{free}}^{(\alpha)} = \sqrt{-g^{(\alpha)}} \; \bar{\psi}_\alpha \left(i \gamma^\mu e_{\ \mu}^{a} \nabla_a - m\right) \psi_\alpha$$

where $e_{\ \mu}^{a}$ is the vierbein and $\nabla_a$ is the spin connection covariant derivative.

### 3.2 Pair Field Lagrangian

$$\mathcal{L}_{\text{pair}}^{(\alpha)} = |D_\mu \Phi_\alpha|^2 - \mu_{\text{pair}}^2 |\Phi_\alpha|^2 - \lambda_{\text{pair}} |\Phi_\alpha|^4 + g_{\text{pair}}\left(\Phi_\alpha^* \, \psi_{\alpha,\uparrow} \psi_{\alpha,\downarrow} + \text{h.c.}\right)$$

**First term:** Kinetic energy of the pair condensate. $D_\mu$ includes electromagnetic coupling with charge $2e$ (the pair carries twice the single-particle charge).

**Second and third terms:** Mexican-hat potential. For $\mu_{\text{pair}}^2 < 0$, the potential has a minimum at:

$$|\Phi_\alpha|^2 = \frac{-\mu_{\text{pair}}^2}{2\lambda_{\text{pair}}} \equiv v_{\text{pair}}^2$$

Spontaneous symmetry breaking of the $U(1)$ phase symmetry yields a massless Goldstone mode (the pair phase $\theta_\alpha$) and a massive radial mode (the Higgs-like amplitude fluctuation) with mass $m_H = \sqrt{-2\mu_{\text{pair}}^2}$.

**Fourth term:** Yukawa coupling between the pair condensate and the underlying fermions. The operator $\psi_{\alpha,\uparrow} \psi_{\alpha,\downarrow}$ creates a pair from two fermions with opposite spin projections. The coupling constant $g_{\text{pair}}$ has dimensions of [mass]$^{-1}$ in 3+1 dimensions.

This Lagrangian is the relativistic generalization of the BCS gap equation. In the non-relativistic limit with a Fermi surface, it reduces to the standard BCS Hamiltonian with gap parameter $\Delta = g_{\text{pair}} \langle \Phi \rangle$.

### 3.3 Transit Lagrangian

$$\mathcal{L}_{\text{transit}} = \sum_{\langle\alpha,\beta\rangle} \left[t_1 \, \bar{\psi}_\alpha \, e^{i\phi_{\alpha\beta}} \, \psi_\beta + t_2 \, \Phi_\alpha^* \, \Phi_\beta + \text{h.c.}\right]$$

This is the inter-universe hopping Lagrangian. It contains two channels:

**Single-particle hopping** ($t_1$ term): A fermion hops from universe $\beta$ to universe $\alpha$. This changes both $N_\alpha$ and $N_\beta$ by $\pm 1$, flipping the parity of both universes. The phase $\phi_{\alpha\beta}$ is a gauge connection on the graph $G_K$ that ensures gauge invariance of the transit amplitude. The hopping amplitude is:

$$t_1 = t_0 \, \exp\left(-\frac{\Delta_E}{k_B T}\right)$$

The exponential suppression by the parity energy gap $\Delta_E$ reflects the energetic cost of flipping parity. At low temperatures, single-particle hopping is exponentially frozen out.

**Pair hopping** ($t_2$ term): A pair (composite boson) hops from universe $\beta$ to universe $\alpha$. This changes both $N_\alpha$ and $N_\beta$ by $\pm 2$, preserving the parity of both universes. Since no parity flip is required:

$$t_2 = t_0$$

The pair hopping amplitude is unsuppressed. This is the dominant transit channel, especially at low temperatures.

The hierarchy $t_2 \gg t_1$ at low temperatures is the microscopic origin of the macroscopic observation that paired systems (superconductors, BECs) maintain coherence while unpaired systems decohere.

### 3.4 Parity Lagrangian

$$\mathcal{L}_{\text{parity}}^{(\alpha)} = -\frac{\Delta}{2} \sigma_\alpha - J \sum_{\beta \sim \alpha} \sigma_\alpha \, \sigma_\beta$$

This is an Ising model on the multiverse graph $G_K$:

**On-site term** ($\Delta/2$): The energy cost of odd parity. $\Delta > 0$ means even parity is energetically preferred. This is the parity energy gap at the single-universe level.

**Coupling term** ($J$): Nearest-neighbor parity interaction. $J > 0$ is ferromagnetic---neighboring universes prefer the same parity. This creates domains of aligned parity across the multiverse.

Although $\sigma_\alpha$ is derived from $N_\alpha$, the Ising representation is useful for mean-field analysis and for identifying the parity phase transition. The mean-field free energy is:

$$F(\langle\sigma\rangle) = -\frac{\Delta}{2}\langle\sigma\rangle - \frac{Jz}{2}\langle\sigma\rangle^2 + \frac{1}{2\beta_{\text{eff}}}\left[\frac{1+\langle\sigma\rangle}{2}\ln\frac{1+\langle\sigma\rangle}{2} + \frac{1-\langle\sigma\rangle}{2}\ln\frac{1-\langle\sigma\rangle}{2}\right]$$

where $\beta_{\text{eff}}$ is an effective inverse temperature governing parity fluctuations.

### 3.5 Gravitational Lagrangian

$$\mathcal{L}_{\text{grav}}^{(\alpha)} = -\frac{1}{16\pi G}\sqrt{-g^{(\alpha)}} \; R^{(\alpha)} + \sqrt{-g^{(\alpha)}} \; T_{\mu\nu}^{(\alpha),\text{paired}} \, h^{(\alpha)\mu\nu}$$

**THE RADICAL TERM.** The first term is the standard Einstein-Hilbert action for the metric in universe $\alpha$. The second term specifies the source: the stress-energy tensor of **paired matter only**.

The modified Einstein field equations are:

$$\boxed{G_{\mu\nu}^{(\alpha)} = 8\pi G \; T_{\mu\nu}^{(\alpha),\text{paired}}}$$

where:

$$T_{\mu\nu}^{(\alpha),\text{paired}} = T_{\mu\nu}^{(\alpha),\text{total}} - T_{\mu\nu}^{(\alpha),\text{unpaired}}$$

Explicitly, the paired stress-energy tensor is constructed from the pair condensate:

$$T_{\mu\nu}^{(\alpha),\text{paired}} = (D_\mu \Phi_\alpha)^*(D_\nu \Phi_\alpha) + (D_\nu \Phi_\alpha)^*(D_\mu \Phi_\alpha) - g_{\mu\nu}^{(\alpha)} \mathcal{L}_{\text{pair}}^{(\alpha)}$$

plus the contribution of fermions that are part of bound pairs (those correlated with the condensate). Unpaired fermions---those in transit between universes---do **not** curve spacetime. This is the central claim connecting parity to gravity.

### 3.6 Anchoring Lagrangian

$$\mathcal{L}_{\text{anchor}}^{(\alpha)} = -\kappa \, A_\alpha(x) \, \bar{\psi}_\alpha \psi_\alpha \, \frac{1 + \sigma_\alpha}{2}$$

This term couples unpaired fermions to the anchoring field (the local density of paired matter). The projection operator $(1 + \sigma_\alpha)/2$ selects the even-parity sector:

$$\frac{1 + \sigma_\alpha}{2} = \begin{cases} 1 & \text{if } \sigma_\alpha = +1 \text{ (even parity)} \\ 0 & \text{if } \sigma_\alpha = -1 \text{ (odd parity)} \end{cases}$$

The physical interpretation: in an even-parity universe, the anchoring field acts as a potential that localizes fermions, suppressing transit. In an odd-parity universe, the anchoring field has no effect---the universe is already quantum-active and particles are free to hop.

The decoherence rate induced by the anchoring field is:

$$\Gamma_{\text{decohere}} = \frac{\kappa \, A_\alpha(x)}{\hbar}$$

Restoring $\hbar$ for physical estimates: near a macroscopic object with $n_{\text{pair}} \sim 10^{23}$ pairs, $\Gamma$ can be extremely large, producing effectively instantaneous decoherence---i.e., classical behavior.

---

## 4. Equations of Motion

The equations of motion are obtained by applying the Euler-Lagrange equations to $\mathcal{L}_{\text{total}}$.

### 4.1 Modified Dirac Equation

Varying $\mathcal{L}_{\text{total}}$ with respect to $\bar{\psi}_\alpha$ yields the modified Dirac equation in universe $\alpha$:

$$\boxed{\left(i\gamma^\mu D_\mu - m\right)\psi_\alpha + g_{\text{pair}} \, \Phi_\alpha \, \psi_{\alpha,c} + t_1 \sum_{\beta \sim \alpha} e^{i\phi_{\alpha\beta}} \psi_\beta - \kappa \, A_\alpha(x) \, \psi_\alpha \, \frac{1 + \sigma_\alpha}{2} = 0}$$

**Term by term:**

1. $(i\gamma^\mu D_\mu - m)\psi_\alpha$: Standard Dirac propagation within universe $\alpha$.

2. $g_{\text{pair}} \, \Phi_\alpha \, \psi_{\alpha,c}$: Pairing interaction. Couples the fermion to the pair condensate via the charge-conjugate field $\psi_c$. This is the Bogoliubov-de Gennes coupling: a fermion can be absorbed into the condensate (paired) or emitted from it.

3. $t_1 \sum_{\beta \sim \alpha} e^{i\phi_{\alpha\beta}} \psi_\beta$: Single-particle hopping from neighboring universes. This is the source of quantum superposition---the fermion amplitude leaks into adjacent universes.

4. $-\kappa \, A_\alpha(x) \, \psi_\alpha \, (1 + \sigma_\alpha)/2$: Anchoring. In even-parity universes with large $A_\alpha$, this acts as an effective mass term that suppresses transit.

In the non-relativistic limit ($E \approx m$, $|\mathbf{p}| \ll m$), the upper two components of $\psi_\alpha$ satisfy a modified Schrodinger equation:

$$i\hbar \frac{\partial}{\partial t}\chi_\alpha = \left[-\frac{\hbar^2}{2m}\nabla^2 + V(x)\right]\chi_\alpha + t_1 \sum_{\beta \sim \alpha} \chi_\beta + g_{\text{pair}} \Delta_{\text{pair}} \chi_\alpha^* - \kappa A_\alpha \chi_\alpha \frac{1+\sigma_\alpha}{2}$$

### 4.2 Pair Field Equation

Varying with respect to $\Phi_\alpha^*$:

$$\boxed{\left(D^\mu D_\mu + \mu_{\text{pair}}^2\right)\Phi_\alpha + 2\lambda_{\text{pair}} |\Phi_\alpha|^2 \Phi_\alpha - g_{\text{pair}} \, \psi_{\alpha,\uparrow} \psi_{\alpha,\downarrow} + t_2 \sum_{\beta \sim \alpha} \Phi_\beta = 0}$$

This is a nonlinear Klein-Gordon equation with:

1. $D^\mu D_\mu \Phi_\alpha$: Wave propagation of pairs within universe $\alpha$.
2. $\mu_{\text{pair}}^2 \Phi_\alpha + 2\lambda_{\text{pair}}|\Phi_\alpha|^2 \Phi_\alpha$: Mexican-hat self-interaction driving spontaneous pairing.
3. $-g_{\text{pair}} \psi_{\uparrow}\psi_{\downarrow}$: Source term---fermions forming pairs feed the condensate.
4. $t_2 \sum_{\beta \sim \alpha} \Phi_\beta$: Pair hopping from neighboring universes. This is the Josephson coupling between universe condensates.

In the static, uniform limit, this reduces to the BCS gap equation:

$$\Delta_{\text{pair}} = g_{\text{pair}}^2 \int_0^{\omega_D} \frac{d\epsilon}{\sqrt{\epsilon^2 + \Delta_{\text{pair}}^2}} \tanh\left(\frac{\sqrt{\epsilon^2 + \Delta_{\text{pair}}^2}}{2k_BT}\right)$$

supplemented by the inter-universe Josephson term $t_2 \sum_\beta \Delta_\beta e^{i\theta_\beta}$.

### 4.3 Parity Dynamics

The parity variable $\sigma_\alpha = (-1)^{N_\alpha}$ is discrete and does not admit a continuous Euler-Lagrange equation. Its dynamics are governed by a master equation (Glauber dynamics for the kinetic Ising model):

$$\boxed{\frac{d}{dt}P(\sigma_\alpha = s, t) = -\Gamma_{\text{flip}}(s) \, P(\sigma_\alpha = s, t) + \Gamma_{\text{flip}}(-s) \, P(\sigma_\alpha = -s, t)}$$

where $s \in \{-1, +1\}$ and the flip rate is:

$$\Gamma_{\text{flip}}(s) = \Gamma_0 \exp\left(-\frac{s}{2k_BT}\left[\Delta + J \sum_{\beta \sim \alpha} \sigma_\beta\right]\right)$$

This is the standard Glauber spin-flip rate for the Ising model. The rate depends on the local parity field---if the neighboring universes are all even ($\sigma_\beta = +1$), flipping from even to odd costs energy $\Delta + Jz$, and the rate is exponentially suppressed.

The mean-field relaxation rate is:

$$\tau_{\text{parity}}^{-1} = \Gamma_0 \left[\exp\left(-\frac{\Delta + Jz\langle\sigma\rangle}{2k_BT}\right) + \exp\left(\frac{\Delta + Jz\langle\sigma\rangle}{2k_BT}\right)\right]$$

At temperatures well below the parity gap, parity flips are exponentially rare, and the system is frozen into a definite parity sector.

### 4.4 Modified Einstein Field Equations

Varying $\mathcal{L}_{\text{grav}}^{(\alpha)}$ with respect to $g^{(\alpha)\mu\nu}$:

$$\boxed{G_{\mu\nu}^{(\alpha)} = 8\pi G \; T_{\mu\nu}^{(\alpha),\text{paired}}}$$

The Bianchi identity $\nabla^\mu G_{\mu\nu} = 0$ requires $\nabla^\mu T_{\mu\nu}^{\text{paired}} = 0$. This is satisfied as long as unpaired fermions carry no gravitational charge---they are absent from the gravitational sector by construction. Within the paired sector, covariant conservation holds by the standard argument.

In the weak-field limit $g_{\mu\nu} = \eta_{\mu\nu} + h_{\mu\nu}$, the linearized equations give the Newtonian potential:

$$\nabla^2 \Phi_{\text{grav}}^{(\alpha)} = 4\pi G \, \rho_{\text{paired}}^{(\alpha)}$$

where $\rho_{\text{paired}}^{(\alpha)} = T_{00}^{(\alpha),\text{paired}}$ is the energy density of paired matter only.

### 4.5 Anchoring Field Equation

The anchoring field $A_\alpha(x)$ is not dynamical. It is determined instantaneously by the distribution of paired matter:

$$\boxed{A_\alpha(x) = \kappa_A \int d^3x' \; K(|x - x'|) \, n_{\text{pair}}(x')}$$

where $n_{\text{pair}}(x') = |\Phi_\alpha(x')|^2$ is the local pair density and $K(r)$ is a spatial kernel. For a Yukawa profile:

$$K(r) = \frac{e^{-r/r_{\text{anchor}}}}{4\pi r_{\text{anchor}}^2 r}$$

The anchoring range $r_{\text{anchor}}$ sets the scale over which classical matter induces decoherence. Near a macroscopic detector ($n_{\text{pair}} \sim 10^{23}$ cm$^{-3}$):

$$A_\alpha \sim \kappa_A \, n_{\text{pair}} \, r_{\text{anchor}}^3$$

This determines the spatial range of the "observation" effect: how close a quantum system must be to a classical apparatus to decohere.

---

## 5. Exact Solutions

### 5.1 $N=1$, $K=2$: The Qubit

The simplest nontrivial system: one particle distributed between two universes. The Hilbert space is spanned by:

$$|1,0\rangle, \quad |0,1\rangle$$

where $|n_1, n_2\rangle$ denotes $n_1$ particles in universe 1 and $n_2$ in universe 2. Both states have one universe odd and one even.

The Hamiltonian restricted to this subspace is:

$$H = \begin{pmatrix} J & t_1 \\ t_1 & J \end{pmatrix}$$

where the diagonal entries are the parity coupling energy (one even-odd pair contributes $-J \cdot (+1)(-1) = J$) and the off-diagonal entries are the single-particle hopping.

**Eigenvalues:**

$$E_{\pm} = J \pm t_1$$

**Eigenstates:**

$$|\pm\rangle = \frac{1}{\sqrt{2}}\left(|1,0\rangle \pm |0,1\rangle\right)$$

**Time evolution:** Starting from $|1,0\rangle$:

$$|\psi(t)\rangle = \cos\left(\frac{t_1 t}{\hbar}\right)|1,0\rangle - i\sin\left(\frac{t_1 t}{\hbar}\right)|0,1\rangle$$

The particle oscillates between universes at angular frequency:

$$\omega = \frac{2t_1}{\hbar}$$

**This is the textbook two-level system.** The Rabi oscillation between two quantum states is exactly inter-universe transit. The "superposition" $|+\rangle$ is a particle equally shared between two universes.

### 5.2 $N=2$, $K=2$: The Pair

Two particles distributed between two universes. The Hilbert space is spanned by:

$$|2,0\rangle, \quad |1,1\rangle, \quad |0,2\rangle$$

The parities are:

| State | $\sigma_1$ | $\sigma_2$ | Parity product |
|---|---|---|---|
| $\|2,0\rangle$ | $+1$ | $+1$ | $+1$ |
| $\|1,1\rangle$ | $-1$ | $-1$ | $+1$ |
| $\|0,2\rangle$ | $+1$ | $+1$ | $+1$ |

Note: $|2,0\rangle$ and $|0,2\rangle$ are both-even (classical). $|1,1\rangle$ is both-odd (quantum-active).

The Hamiltonian is:

$$H = \begin{pmatrix} -\Delta - J & \sqrt{2}\,t_1 & t_2 \\ \sqrt{2}\,t_1 & \Delta - J & \sqrt{2}\,t_1 \\ t_2 & \sqrt{2}\,t_1 & -\Delta - J \end{pmatrix}$$

**Diagonal entries:**

- $|2,0\rangle$ and $|0,2\rangle$: Both universes even, so parity on-site energy is $-\Delta$ (even preferred). Coupling: $\sigma_1 \sigma_2 = (+1)(+1) = +1$, contributing $-J$.
- $|1,1\rangle$: Both universes odd, so parity energy is $+\Delta$. Coupling: $\sigma_1 \sigma_2 = (-1)(-1) = +1$, contributing $-J$.

**Off-diagonal entries:**

- $\langle 2,0|H|1,1\rangle = \sqrt{2}\,t_1$: One particle hops, factor $\sqrt{2}$ from bosonic enhancement (two equivalent single-particle hops).
- $\langle 2,0|H|0,2\rangle = t_2$: Both particles hop together as a pair.

**In the regime $\Delta \gg t_1$ (strong parity gap):**

The $|1,1\rangle$ state is lifted in energy by $2\Delta$ relative to $|2,0\rangle$ and $|0,2\rangle$. It is energetically frozen out. The effective $2 \times 2$ Hamiltonian in the $\{|2,0\rangle, |0,2\rangle\}$ subspace is:

$$H_{\text{eff}} = \begin{pmatrix} -\Delta - J + \frac{2t_1^2}{2\Delta} & t_2 + \frac{2t_1^2}{2\Delta} \\ t_2 + \frac{2t_1^2}{2\Delta} & -\Delta - J + \frac{2t_1^2}{2\Delta} \end{pmatrix}$$

where the $t_1^2/(2\Delta)$ terms arise from second-order virtual hopping through $|1,1\rangle$ (Schrieffer-Wolff transformation).

The pair oscillates between $|2,0\rangle$ and $|0,2\rangle$ at frequency:

$$\omega_{\text{pair}} = \frac{2}{\hbar}\left(t_2 + \frac{t_1^2}{\Delta}\right) \approx \frac{2t_2}{\hbar}$$

since $t_2 \gg t_1^2/\Delta$. The pair hops as a unit, maintaining even parity in both universes throughout.

### 5.3 $N=3$, $K=2$: First Odd Total

Three particles in two universes. The Hilbert space:

$$|3,0\rangle, \quad |2,1\rangle, \quad |1,2\rangle, \quad |0,3\rangle$$

Parities:

| State | $\sigma_1$ | $\sigma_2$ | Total parity $\sigma_{\text{tot}}$ |
|---|---|---|---|
| $\|3,0\rangle$ | $-1$ | $+1$ | $-1$ |
| $\|2,1\rangle$ | $+1$ | $-1$ | $-1$ |
| $\|1,2\rangle$ | $-1$ | $+1$ | $-1$ |
| $\|0,3\rangle$ | $+1$ | $-1$ | $-1$ |

**Crucially:** $N_{\text{tot}} = 3$ is odd, so $\sigma_{\text{tot}} = -1$. At least one universe is **always** odd. The system can never reach a state where both universes are even. Full pairing is impossible.

The $4 \times 4$ Hamiltonian is:

$$H = \begin{pmatrix} \Delta + J & \sqrt{3}\,t_1 & t_2\sqrt{3} & 0 \\ \sqrt{3}\,t_1 & -\Delta + J & 2t_1 & t_2\sqrt{3} \\ t_2\sqrt{3} & 2t_1 & \Delta + J & \sqrt{3}\,t_1 \\ 0 & t_2\sqrt{3} & \sqrt{3}\,t_1 & -\Delta + J \end{pmatrix}$$

where the off-diagonal elements include the appropriate combinatorial factors for fermionic hopping ($\sqrt{n}$ factors) and pair hopping.

**Key physics:** The states $|3,0\rangle$ and $|1,2\rangle$ have universe 1 odd; the states $|2,1\rangle$ and $|0,3\rangle$ have universe 2 odd. There is always one odd universe, so quantum behavior (inter-universe transit) persists indefinitely. No anchoring mechanism can fully classicalize this system.

This is the microscopic realization of the paper's central claim: **the universe must be odd for quantum mechanics to operate.** An odd total particle number is the necessary condition for permanent quantum activity.

### 5.4 General $N$, $K=2$: Even/Odd Dichotomy

For general $N$ particles in two universes, the Hilbert space is $(N+1)$-dimensional: $\{|N-k, k\rangle : k = 0, 1, \ldots, N\}$.

**$N$ even:** The state $|N/2, N/2\rangle$ has both universes with the same parity ($N/2$ each). If $N/2$ is even, both are in the even (classical) sector. The system can reach a fully paired ground state.

**$N$ odd:** No state has both universes even. At least one universe is always odd. Quantum transit never fully freezes out.

The energy gap between the lowest even-parity state and the lowest odd-parity state is:

$$\Delta E_{\text{even-odd}} = \Delta + 2J + O(t_1^2/\Delta)$$

This gap determines the stability of the classical regime for even-$N$ systems.

---

## 6. Key Derived Quantities

### 6.1 Parity Energy Gap

The parity energy gap is the energy difference between odd-parity and even-parity configurations of a universe:

$$\boxed{\Delta_E = E(\text{odd}) - E(\text{even}) = \Delta + 2Jz\langle\sigma\rangle}$$

In mean-field theory, the expectation value $\langle\sigma\rangle$ satisfies the self-consistency equation:

$$\langle\sigma\rangle = \tanh\left(\beta_{\text{eff}}\left[\Delta + Jz\langle\sigma\rangle\right]\right)$$

where $\beta_{\text{eff}} = 1/(k_BT_{\text{eff}})$ is an effective inverse temperature governing parity fluctuations.

This equation has a phase transition at:

$$\beta_{\text{eff}} \cdot J \cdot z = 1$$

Below this critical temperature, the multiverse spontaneously orders into a parity-aligned phase ($\langle\sigma\rangle \neq 0$). Above it, parity is disordered.

**Estimated value:** From matching decoherence rates in existing experiments, we estimate:

$$\Delta_E \sim 10^{-6} \text{ to } 10^{-3} \text{ eV}$$

The lower bound comes from the requirement that mesoscopic systems ($N \sim 10^6$) show coherence; the upper bound from the requirement that macroscopic systems ($N \sim 10^{23}$) decohere effectively instantaneously.

### 6.2 Even/Odd Coherence Time Ratio

This is the **central testable prediction** of the theory. In a system with controllable particle number, the ratio of coherence times for even-$N$ versus odd-$N$ configurations is:

$$\boxed{\frac{\tau_{\text{even}}}{\tau_{\text{odd}}} = \exp\left(\frac{\Delta_E}{k_BT}\right)}$$

The even-parity system has a gap $\Delta_E$ protecting it from parity-flipping decoherence events. The odd-parity system has no such protection (it is already in the quantum-active sector).

**Numerical estimates:**

| Temperature | $\Delta_E$ | Ratio $\tau_{\text{even}}/\tau_{\text{odd}}$ |
|---|---|---|
| $T = 100$ mK | $10^{-4}$ eV | $\exp(11.6) \approx 1.1 \times 10^5$ |
| $T = 100$ mK | $10^{-5}$ eV | $\exp(1.16) \approx 3.2$ |
| $T = 300$ K | $10^{-4}$ eV | $\exp(3.87) \approx 48$ |
| $T = 300$ K | $10^{-3}$ eV | $\exp(38.7) \approx 6.2 \times 10^{16}$ |
| $T = 10$ mK | $10^{-4}$ eV | $\exp(116) \approx 10^{50}$ |

At millikelvin temperatures with $\Delta_E \sim 10^{-4}$ eV, the ratio exceeds $10^4$---a dramatic, unambiguous signal. At room temperature, even a modest gap of $10^{-4}$ eV predicts a factor of ~50 difference, which is experimentally detectable in cold-atom or ion-trap systems.

### 6.3 Effective Gravitational Mass

Since only paired matter sources gravity, the gravitational mass of a particle in a multiverse superposition state $|\psi\rangle = \sum_\alpha c_\alpha |\alpha\rangle$ is:

$$\boxed{m_{\text{grav}}^{(\alpha)} = m \, |c_\alpha|^2 \, f_{\text{pair}}^{(\alpha)}}$$

where $f_{\text{pair}}^{(\alpha)} = N_\alpha^{\text{pair}} / (N_\alpha/2)$ is the pairing fraction in universe $\alpha$ (the fraction of fermions that are paired, normalized so that $f_{\text{pair}} = 1$ when all fermions are paired).

The total gravitational mass summed over all universes:

$$m_{\text{grav}}^{\text{total}} = m \sum_{\alpha=1}^{K} |c_\alpha|^2 \, f_{\text{pair}}^{(\alpha)} = m \langle f_{\text{pair}} \rangle$$

**Limiting cases:**

- **Fully paired** ($f_{\text{pair}} = 1$, all particles anchored): $m_{\text{grav}} = m$. Standard gravity recovered.
- **Fully unpaired in transit** ($f_{\text{pair}} = 0$, particle delocalized across universes): $m_{\text{grav}} = 0$. No gravitational interaction.
- **Macroscopic object** ($N \sim 10^{23}$, one unpaired particle): $f_{\text{pair}} = 1 - 1/N \approx 1$. Negligible deviation from standard gravity.

### 6.4 Critical Pairing Fraction

The critical pairing fraction at which the system transitions from quantum-active to classical-dominant is:

$$\boxed{f_c = 1 - \frac{1}{2N}}$$

For $N$ particles, the system is in the even-parity (classical) sector when all but at most zero particles are paired. Since pairing consumes particles in groups of 2, the last unpaired particle (for odd $N$) prevents $f_{\text{pair}}$ from reaching 1.

The transition is **sharp** (first-order-like) because parity is binary---there is no continuous interpolation between $\sigma = +1$ and $\sigma = -1$. The crossover width in $f_{\text{pair}}$ is determined by thermal fluctuations:

$$\delta f \sim \sqrt{\frac{k_BT}{N \cdot \Delta_E}}$$

For $N = 10^{6}$, $T = 1$ K, $\Delta_E = 10^{-4}$ eV: $\delta f \sim 10^{-4}$. The transition is extremely sharp even for mesoscopic systems.

### 6.5 Hopping Ratio

The ratio of pair hopping to single-particle hopping amplitudes:

$$\boxed{\frac{t_2}{t_1} = \exp\left(\frac{\Delta_E}{k_BT}\right)}$$

This ratio controls the dominant transport mechanism:

| Temperature | $\Delta_E$ | $t_2/t_1$ |
|---|---|---|
| $T = 300$ K | $10^{-4}$ eV | $\approx 48$ |
| $T = 100$ mK | $10^{-4}$ eV | $\approx 1.1 \times 10^5$ |
| $T = 10$ mK | $10^{-4}$ eV | $\approx 10^{50}$ |
| $T = 1$ mK | $10^{-4}$ eV | $\approx 10^{503}$ |

At millikelvin temperatures, single-particle hopping is effectively impossible. Only pairs can transit. This is why superconductors and BECs exhibit macroscopic coherence: the only available transit mode (pair hopping) preserves parity and thus preserves the coherent condensate.

At room temperature, single-particle hopping is merely suppressed by a factor of ~50, explaining why quantum effects are observable but fragile at everyday temperatures.

---

## 7. Consistency Checks

### 7.1 Lorentz Invariance

Each universe $\alpha$ is independently Lorentz covariant. The free Dirac Lagrangian $\mathcal{L}_{\text{free}}^{(\alpha)}$ and pair Lagrangian $\mathcal{L}_{\text{pair}}^{(\alpha)}$ are manifestly Lorentz invariant within universe $\alpha$.

The transit Lagrangian $\mathcal{L}_{\text{transit}}$ contains terms of the form $\bar{\psi}_\alpha \psi_\beta$ and $\Phi_\alpha^* \Phi_\beta$. These are Lorentz **scalars** constructed from fields in different universes. Since the universes share the same Lorentz group structure (same metric signature, same Clifford algebra), the bilinear $\bar{\psi}_\alpha \psi_\beta$ transforms as a scalar under simultaneous Lorentz transformations in both universes.

There is no preferred frame within any single universe. The multiverse graph $G_K$ defines a structure in universe-label space, not in spacetime. Lorentz invariance is preserved.

### 7.2 Energy Conservation

The total multiverse action $S = \int d^4x \; \mathcal{L}_{\text{total}}$ is invariant under time translations $t \to t + \epsilon$ (all universes share the same time parameter). By Noether's theorem, the total multiverse energy is conserved:

$$E_{\text{total}} = \sum_{\alpha=1}^{K} E_\alpha = \text{const}$$

Individual universe energies $E_\alpha$ are **not** conserved---particles transit between universes, carrying energy with them. This is entirely analogous to coupled subsystems in ordinary mechanics: the energy of each subsystem fluctuates, but the total is conserved.

The transit Lagrangian explicitly mediates energy exchange:

$$\frac{dE_\alpha}{dt} = t_1 \sum_{\beta \sim \alpha} \text{Re}\left(\bar{\psi}_\alpha \psi_\beta\right) + t_2 \sum_{\beta \sim \alpha} \text{Re}\left(\Phi_\alpha^* \, \partial_t \Phi_\beta\right) \neq 0$$

### 7.3 Weak Equivalence Principle

The weak equivalence principle (WEP) states that all objects fall at the same rate regardless of composition: $m_{\text{grav}} / m_{\text{inertial}} = 1$.

In our framework, this ratio is:

$$\frac{m_{\text{grav}}}{m_{\text{inertial}}} = f_{\text{pair}}$$

For unpaired matter, $f_{\text{pair}} < 1$, and the WEP is **violated**. The Eotvos parameter measuring the violation is:

$$\eta_{\text{WEP}} = 1 - f_{\text{pair}}$$

**For macroscopic matter:** A macroscopic object of mass $M$ containing $N \sim 10^{23}$ fermions has at most a few unpaired particles (if total particle number is odd, exactly one is unpaired). Therefore:

$$f_{\text{pair}} = 1 - \frac{N_{\text{unpaired}}}{N} \geq 1 - \frac{1}{N} \approx 1 - 10^{-23}$$

$$\eta_{\text{WEP}} \leq 10^{-23}$$

Current Eotvos experiments (MICROSCOPE satellite, 2022) achieve sensitivity $\eta < 10^{-15}$. The predicted violation of $\eta \sim 10^{-23}$ for macroscopic objects is eight orders of magnitude below detection threshold. The WEP is recovered as an emergent property of macroscopic pairing, with violations detectable only in proposed mesoscopic superposition experiments where $f_{\text{pair}}$ is significantly less than 1.

**For mesoscopic systems:** A nanoparticle with $N \sim 10^{6}$ atoms in a superposition state might have $f_{\text{pair}} \sim 0.5$, giving $\eta_{\text{WEP}} \sim 0.5$---a dramatic violation detectable by proposed experiments (MAQRO, optomechanical resonators).

### 7.4 Single-Universe Limit ($K = 1$)

When $K = 1$, the multiverse graph has a single vertex and no edges:

- $\mathcal{L}_{\text{transit}} = 0$: No inter-universe hopping.
- $\sigma_\alpha$ is fixed: No parity dynamics (no neighbors to couple to).
- $\mathcal{L}_{\text{parity}} = -\Delta \sigma / 2$: A constant energy shift.
- $\mathcal{L}_{\text{pair}}$ retains its structure: Pairing still occurs within the universe.

The theory reduces to standard quantum field theory in a single universe with a BCS-like pairing interaction. There is no mechanism for superposition-as-transit, no inter-universe interference. This confirms that $K \geq 2$ is required for the parity framework to produce quantum phenomena.

### 7.5 Standard Quantum Mechanics Recovery ($K \to \infty$)

This is the crucial consistency check. On a regular $d$-dimensional lattice with spacing $a$, the hopping sum becomes a discrete Laplacian:

$$t_1 \sum_{\beta \sim \alpha} \psi_\beta = t_1 \left(\sum_{\beta \sim \alpha} \psi_\beta - z \, \psi_\alpha\right) + z \, t_1 \, \psi_\alpha = t_1 \, (\Delta_G \psi)_\alpha + z \, t_1 \, \psi_\alpha$$

In the continuum limit $a \to 0$, $K \to \infty$ with:

$$t_1 = \frac{\hbar^2}{2m a^2}$$

the discrete Laplacian becomes the continuous Laplacian:

$$t_1 \, (\Delta_G \psi)_\alpha \;\longrightarrow\; \frac{\hbar^2}{2m} \nabla^2 \psi(x)$$

where we identify the universe label $\alpha$ with a position $x$ in configuration space. The modified Dirac equation (Section 4.1), in the non-relativistic limit with $\Delta, J, \kappa \to 0$ (no parity, no anchoring), becomes:

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2 \psi + V(x)\psi$$

**This is the Schrodinger equation.** The configuration space of quantum mechanics is identified with the universe-label space of the multiverse. A "wavefunction" $\psi(x)$ is the amplitude for a particle to be at position $x$ in universe-label space. This confirms the Many-Interacting-Worlds mechanism of Hall, Deckert, and Wiseman (2014).

### 7.6 MIW Recovery

Setting the parity-specific parameters to zero:

$$\Delta = 0, \quad J = 0, \quad t_1 = t_2 \equiv t, \quad \kappa = 0$$

eliminates all parity structure (no energy gap, no parity coupling, equal hopping for singles and pairs, no anchoring). The Lagrangian reduces to:

$$\mathcal{L} = \sum_\alpha \mathcal{L}_{\text{free}}^{(\alpha)} + t \sum_{\langle\alpha,\beta\rangle} \left(\bar{\psi}_\alpha \psi_\beta + \Phi_\alpha^* \Phi_\beta + \text{h.c.}\right)$$

This is exactly the MIW Hamiltonian: many classical worlds interacting through a universal hopping parameter. The parity framework is a strict extension of MIW, reducing to it when all parity-related parameters vanish.

---

## 8. Connection to Existing Frameworks

### 8.1 BCS Superconductivity

The pair Lagrangian $\mathcal{L}_{\text{pair}}$ (Section 3.2) is the relativistic generalization of BCS theory, extended to the multiverse:

| BCS Concept | Parity Framework Analog |
|---|---|
| Cooper pair in metal | Paired fermions in universe $\alpha$ |
| BCS gap $\Delta$ | $\|\langle\Phi_\alpha\rangle\|$ |
| Normal state (above $T_c$) | Odd-parity sector ($\sigma = -1$) |
| Superconducting state (below $T_c$) | Even-parity sector ($\sigma = +1$) |
| Josephson junction coupling | Inter-universe pair hopping $t_2$ |
| Quasiparticle excitation | Unpaired fermion in transit |

The BCS gap equation (Section 4.2) is recovered in the static, single-universe limit. The inter-universe Josephson coupling $t_2$ is a new prediction: pair condensates in different universes are phase-coherent.

### 8.2 Z2 Lattice Gauge Theory

The parity sector of the theory has the structure of a $\mathbb{Z}_2$ lattice gauge theory on $G_K$:

- **Matter fields** $\psi_\alpha$: Live on vertices (universes).
- **Gauge links** $\sigma_{\alpha\beta} = \sigma_\alpha \sigma_\beta$: Live on edges (transit channels).
- **Wilson loop** around a cycle $C$ in $G_K$:

$$W(C) = \prod_{\langle\alpha,\beta\rangle \in C} \sigma_\alpha \sigma_\beta = \prod_{\alpha \in C} \sigma_\alpha^{2} = 1 \text{ (trivially)}$$

For the Ising model, the Wilson loop is always 1 because $\sigma_\alpha^2 = 1$. However, the **frustrated Wilson loop** on non-bipartite graphs can detect topological obstructions to parity alignment:

$$W_F(C) = \prod_{\langle\alpha,\beta\rangle \in C} J_{\alpha\beta} \, \sigma_\alpha \sigma_\beta$$

where $J_{\alpha\beta}$ can be negative (antiferromagnetic) on some edges.

The correspondence to gauge theory phases:

- **Confinement** ($\langle\sigma\rangle \neq 0$, ordered phase): Classical regime. Parity is frozen, particles are anchored.
- **Deconfinement** ($\langle\sigma\rangle = 0$, disordered phase): Quantum regime. Parity fluctuates freely, particles transit between universes.

### 8.3 Many-Interacting-Worlds (MIW)

Our framework extends MIW in four specific directions:

1. **Parity structure:** The $\mathbb{Z}_2$ order parameter $\sigma_\alpha$ and the Ising Lagrangian $\mathcal{L}_{\text{parity}}$ provide an energetic distinction between quantum-active and classical-dominant universes. MIW treats all worlds equivalently.

2. **Single/pair distinction:** The hierarchy $t_2 \gg t_1$ at low temperatures introduces a qualitative difference between single-particle and pair-mediated hopping. MIW has a single hopping parameter.

3. **Modified gravity:** The restriction $G_{\mu\nu} = 8\pi G \, T_{\mu\nu}^{\text{paired}}$ is absent in MIW, which does not address gravity.

4. **Anchoring mechanism:** The anchoring Lagrangian $\mathcal{L}_{\text{anchor}}$ provides a decoherence mechanism tied to the local density of classical matter. MIW does not include decoherence.

When all four extensions are removed ($\Delta = J = 0$, $t_1 = t_2$, $\kappa = 0$, standard gravity), we recover MIW exactly (Section 7.6).

### 8.4 Diosi-Penrose Objective Collapse

Diosi (1987) and Penrose (1996) independently proposed that gravity causes wavefunction collapse on a timescale:

$$\tau_{\text{DP}} = \frac{\hbar \, d}{G \, m^2}$$

where $d$ is the spatial separation of the superposed states and $m$ is the mass.

Our framework makes a sharply different prediction: **unpaired matter does not source gravity** ($m_{\text{grav}} = 0$ for $f_{\text{pair}} = 0$). Therefore, there is no gravitational self-energy to drive collapse. A massive particle in a genuine quantum superposition (fully unpaired, in transit between universes) experiences no gravitational decoherence whatsoever.

The **distinguishing experiment** is a massive isolated superposition:

| Prediction | Diosi-Penrose | Parity Framework |
|---|---|---|
| Superposition lifetime | $\tau_{\text{DP}} = \hbar d / (Gm^2)$ | No gravitational collapse; limited by $\kappa \cdot A$ (anchoring) |
| Mass dependence | Collapse faster for larger $m$ | No mass dependence for isolated systems |
| Isolation effect | Irrelevant (gravity is self-interaction) | Reduces $A(x)$, extends coherence |

If a massive ($m > 10^{-14}$ kg) object can be placed in spatial superposition in a well-isolated environment, and the superposition persists beyond $\tau_{\text{DP}}$, Diosi-Penrose is falsified and the parity framework is supported.

---

## 9. Free Parameters and Constraints

### 9.1 Parameter Inventory

The theory contains the following parameters:

| Parameter | Symbol | Description | Dimensions |
|---|---|---|---|
| Base hopping amplitude | $t_0$ | Intrinsic transit rate | [energy] |
| Parity on-site energy | $\Delta$ | Cost of odd parity | [energy] |
| Parity coupling | $J$ | Nearest-neighbor Ising coupling | [energy] |
| Pairing Yukawa coupling | $g_{\text{pair}}$ | Fermion-pair coupling | [mass]$^{-1}$ |
| Pair potential mass | $\mu_{\text{pair}}$ | Mexican-hat parameter | [mass] |
| Pair self-coupling | $\lambda_{\text{pair}}$ | Quartic coupling | [dimensionless] |
| Anchoring coupling | $\kappa$ | Decoherence strength | [energy $\cdot$ length$^3$] |
| Anchoring range | $r_{\text{anchor}}$ | Spatial scale of anchoring | [length] |
| Number of universes | $K$ | Graph size | [dimensionless] |

### 9.2 Constraints

**From Schrodinger equation recovery (Section 7.5):**

$$t_0 = \frac{\hbar^2}{2m a^2}$$

This fixes $t_0$ in terms of the lattice spacing $a$ and particle mass $m$. In the continuum limit, $t_0 \to \infty$ and $a \to 0$ with $t_0 a^2$ fixed.

**From BCS-scale gap:**

$$g_{\text{pair}}, \mu_{\text{pair}}, \lambda_{\text{pair}}$$ are constrained to reproduce the known BCS gap $\Delta_{\text{BCS}} \sim 10^{-3}$ eV in superconductors. This fixes the ratios $g_{\text{pair}}^2 / \lambda_{\text{pair}}$ and $\mu_{\text{pair}}^2 / \lambda_{\text{pair}}$.

**From known decoherence rates:**

The product $\kappa \cdot A$ must reproduce observed decoherence rates in mesoscopic systems:

$$\Gamma_{\text{decohere}} = \frac{\kappa \cdot A}{\hbar} \sim 10^{6} \text{ to } 10^{12} \text{ s}^{-1}$$

for typical laboratory environments. This constrains $\kappa \cdot \kappa_A \cdot n_{\text{pair}} \cdot r_{\text{anchor}}^3$.

**From WEP bounds:**

The pairing fraction for macroscopic matter must satisfy $f_{\text{pair}} > 1 - 10^{-15}$, which is trivially satisfied for $N > 10^{15}$.

**From Arndt et al. (1999), Fein et al. (2019) interference experiments:**

Large-molecule interference ($m \sim 10^{-22}$ kg for C$_{60}$, $m \sim 10^{-21}$ kg for 2000-atom molecules) constrains the anchoring range:

$$r_{\text{anchor}} < d_{\text{slit-screen}} \sim 1 \text{ m}$$

and the anchoring coupling must be weak enough that molecules in vacuum maintain coherence over the experimental length scale.

### 9.3 Truly Free Parameters

After applying all constraints, the theory has **two** genuinely free parameters:

$$\boxed{\Delta_E \text{ (parity energy gap)} \qquad \text{and} \qquad \kappa \text{ (anchoring coupling)}}$$

These two parameters determine all testable predictions:

- $\Delta_E$ sets the even/odd coherence time ratio (Section 6.2), the hopping ratio (Section 6.5), and the parity phase transition temperature.
- $\kappa$ sets the decoherence rate near classical matter (Section 3.6) and the spatial range of the "observation" effect.

A single experiment measuring parity-dependent decoherence (Section 6.2) at two different temperatures would determine both $\Delta_E$ (from the ratio) and $\kappa$ (from the absolute rate), yielding a fully predictive theory with zero free parameters.

---

## 10. Summary of Testable Predictions

For convenience, we collect the quantitative predictions that distinguish this framework from standard quantum mechanics and from competing interpretations:

| Prediction | Formula | Key dependence |
|---|---|---|
| Even/odd coherence ratio | $\tau_{\text{even}}/\tau_{\text{odd}} = \exp(\Delta_E / k_BT)$ | Exponential in $\Delta_E/T$ |
| Gravitational mass of superposed matter | $m_{\text{grav}} = m \cdot f_{\text{pair}}$ | Vanishes for unpaired matter |
| WEP violation for mesoscopic systems | $\eta = 1 - f_{\text{pair}}$ | Detectable for $N < 10^{15}$ |
| No gravitational collapse of superpositions | $\tau > \tau_{\text{DP}}$ for isolated systems | Contradicts Diosi-Penrose |
| Pair hopping dominance at low $T$ | $t_2/t_1 = \exp(\Delta_E / k_BT)$ | Explains BEC/superconductor coherence |
| Parity phase transition | $T_c = Jz / k_B$ | Multiverse-scale ordering |

---

## References

1. Arndt, M., Nairz, O., Vos-Andreae, J., Keller, C., van der Zouw, G., and Zeilinger, A. (1999). "Wave-particle duality of C60 molecules." *Nature*, 401(6754), 680--682.

2. Bardeen, J., Cooper, L.N., and Schrieffer, J.R. (1957). "Theory of superconductivity." *Physical Review*, 108(5), 1175--1204.

3. Diosi, L. (1987). "A universal master equation for the gravitational violation of quantum mechanics." *Physics Letters A*, 120(8), 377--381.

4. Fein, Y.Y., Geyer, P., Zwick, P., Kialka, F., Pedalino, S., Mayor, M., Gerlich, S., and Arndt, M. (2019). "Quantum superposition of molecules beyond 25 kDa." *Nature Physics*, 15(12), 1242--1245.

5. Hall, M.J.W., Deckert, D.-A., and Wiseman, H.M. (2014). "Quantum phenomena modeled by interactions between many classical worlds." *Physical Review X*, 4(4), 041013.

6. Itano, W.M., Heinzen, D.J., Bollinger, J.J., and Wineland, D.J. (1990). "Quantum Zeno effect." *Physical Review A*, 41(5), 2295--2300.

7. Jacques, V., Wu, E., Grosshans, F., Treussart, F., Grangier, P., Aspect, A., and Roch, J.-F. (2007). "Experimental realization of Wheeler's delayed-choice gedanken experiment." *Science*, 315(5814), 966--968.

8. Kitaev, A.Y. (2003). "Fault-tolerant quantum computation by anyons." *Annals of Physics*, 303(1), 2--30.

9. Misra, B. and Sudarshan, E.C.G. (1977). "The Zeno's paradox in quantum theory." *Journal of Mathematical Physics*, 18(4), 756--763.

10. Parker, L. (1968). "Particle creation in expanding universes." *Physical Review Letters*, 21(8), 562--564.

11. Penrose, R. (1996). "On gravity's role in quantum state reduction." *General Relativity and Gravitation*, 28(5), 581--600.

12. Sakharov, A.D. (1967). "Violation of CP invariance, C asymmetry, and baryon asymmetry of the universe." *JETP Letters*, 5, 24--27.

13. Shor, P.W. (1995). "Scheme for reducing decoherence in quantum computer memory." *Physical Review A*, 52(4), R2493--R2496.

14. Wick, G.C., Wightman, A.S., and Wigner, E.P. (1952). "The intrinsic parity of elementary particles." *Physical Review*, 88(1), 101--105.

15. Witten, E. (1982). "Constraints on supersymmetry breaking." *Nuclear Physics B*, 202(2), 253--316.

---

*AI assistance from Claude (Anthropic) was used for mathematical verification and structural organization. All physical hypotheses and theoretical claims originate from the human author.*
