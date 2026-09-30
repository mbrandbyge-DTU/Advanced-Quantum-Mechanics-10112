# Lecture activities and simple student edits

These activities accompany the time-dependent-phenomena lecture scheduled for
5 October 2026 in Overview2026.pdf (Ballentine pp. 332–349). The course plan lists
stationary perturbation theory on 1 October. The notebooks also support the
following time-dependent-perturbation lecture on 8 October.

## A practical lecture rhythm

Use three short cycles of **predict individually → discuss in pairs → run → explain**.
Ask students to write a prediction before touching a slider. One student operates
the notebook and the other checks the physics; swap roles after each experiment.
Allow about 5 minutes per cycle, distributed across the lecture.

1. **Before spin precession:** ask what changes for +z versus +x in a static z field.
   Have students name an observable that changes and one that stays fixed.
2. **Before the resonance formula:** start with +z and a transverse circular field.
   Ask pairs to find complete inversion and explain the signed drive frequency.
   Then detune and predict both oscillation period and maximum flip probability.
3. **Before exponential decay and “watching”:** start at site 1, compare two sites
   with a long chain, then zoom into early times. Ask whether a unitary system
   can have a decaying local population. Introduce the green measurement curve
   only after students have recognized the quadratic short-time law.

The full future trajectory is shown in the probability panel. For prediction
activities, have students commit their prediction before opening the experiment
or before changing its parameters.

## Three quick concept checks

These can be asked by a show of hands, your existing polling system, or a short
written response. They need no new accounts or in-notebook AI connection.

**A. Static spin**
For initial +x in a static z field, which statement is correct?

- A: P(+z) oscillates.
- B: P(+z) stays 1/2, while the relative phase changes.
- C: The state is completely time independent.

Answer: B. Ask a student to identify a changing expectation value.

**B. Detuned circular drive**
Starting at +z, detuning a circular drive from resonance:

- A: reduces the maximum inversion and increases the generalized Rabi frequency.
- B: reduces the maximum inversion and decreases the generalized Rabi frequency.
- C: leaves maximum inversion equal to one.

Answer: A. The effective field tilts away from the transverse plane and its
magnitude increases at fixed drive amplitude.

**C. Quantum Zeno**
Which ingredient causes the ideal Zeno effect in this notebook?

- A: Drawing the probability more frequently.
- B: Ideal repeated projective tests of the initial state.
- C: Total probability leaking out of the Hilbert space.

Answer: B. Ask students what happens to the normalized state after a YES and
what probability is represented by the all-YES curve.

## First code edits — 5–10 minutes each

Start with changes that require one physics function and no UI changes.

### Chain: a scattering site

In `chain_hamiltonian`, immediately before `return H`, add:

```python
H[N//2, N//2] = 2.0
```

Use a Gaussian in a uniform 100-site chain, centred around site 20, width 3,
k/pi = 0.5. Predict reflection/transmission. Change the barrier height and explain
why the return probability need not equal the occupation of the initial centre site.
Check norm and energy conservation using the diagnostic line below the plot.

### Chain: turn the open chain into a ring

Set g = 1 and epsilon = 0, and use N >= 3. Before `return H`, add:

```python
H[0, -1] = H[-1, 0] = -1.0
```

Predict spreading in both directions from site 1. Compare interference and
recurrences with the open chain. Do not apply this edit to N = 2: it would
overwrite an existing bond instead of adding a new one.

### Spin: a new initial state

In the `initial` dropdown add the string `"-x"`. In the `_angles` dictionary
in the control-to-physics cell add:

```python
"-x": (np.pi/2, np.pi)
```

Explain the initial amplitudes and verify that the Bloch vector is (-1,0,0).
This exercise touches the initial-state mapping, not the plotting code.

### Spin: verify the exact solution independently

For stronger students, use a numerical solution of `i*dpsi/dt = H(t)*psi` for the
same circular field. Compare amplitudes, not only probabilities. Converge the
time step or integrator tolerance. Only then replace the circular field by a
linearly polarized field; the constant rotating-frame Hamiltonian in the notebook
will no longer be an exact solution. In the weak-drive rotating-wave limit, a
linear drive with the same cosine coefficient has half the resonant transverse
strength of the circular drive used here.

## Use AI after the student has made a physics commitment

A simple approach is to use students' existing AI access. No API key, embedded
chatbot, or AI dependency is required by these notebooks.

- **Sign check:** “Here are my Hamiltonian, rotating transformation, and derivation.
  Identify the first incorrect step, if any, and explain it without rewriting the
  whole derivation.” Students must verify the result with signed resonance.
- **Short-time reasoning:** “Here is my expansion of the return amplitude through
  t^2. Check the probability expansion and tell me why there is no linear term.”
  Students must connect the answer to the plotted variance of energy.
- **Minimal code help:** “Modify only `chain_hamiltonian` to add an onsite barrier.
  Explain the changed matrix elements. Do not alter the controls or plotting.”
  Students inspect the matrix and check conservation before accepting the change.
- **Critique an overclaim:** “An AI says a finite Hermitian chain gives exact
  exponential decay for all times. Give two numerical or analytic tests that
  can falsify this claim.” Require both the initial slope and finite-size return.

For formative feedback, ask each pair to submit: **one prediction, one plotted
observation, one equation, and one AI claim they checked**. A later instructor
workflow could use AI to draft new distractors from these responses, with your
review before showing them to the class.

## Instructor physics notes

### Spin conventions

The spin code preserves the slides' minus sign:
H(t) = -[omega_0 sigma_z + omega_1(cos(omega t) sigma_x + sin(omega t) sigma_y)]/2.
With R(t) = exp(-i omega t sigma_z/2), H_eff = -[(omega_0+omega)sigma_z +
omega_1 sigma_x]/2. Resonance is omega = -omega_0. Omega symbols are angular
frequencies, not frequencies in Hz; omega_i = gamma B_i with signed gamma.
The green arrow shows the normalized frequency vector, or its effective
rotating-frame counterpart, rather than a calibrated magnetic field.

P(-z) has the same value in both frames. For initial +z, its maximum is
omega_1^2 / [(omega_0+omega)^2 + omega_1^2]. On resonance the first full flip is
at pi / |omega_1|. A pure-state Bloch vector has unit length; it is twice the
spin expectation in hbar = 1 units. The notebook models no relaxation.

### Chain and exponential comparison

The chain preserves the Mathematica first-site energy epsilon and a special
first hopping beta, represented here as beta = -g. The remaining hopping is -1.
For a site-1 initial state, the short-time energy variance is g^2.
For a semi-infinite uniform chain the end-site density of states is
rho_end(E) = sqrt(4-E^2)/(2*pi), inside the band [-2,2]. Therefore the weak-coupling
probability decay-rate estimate is Gamma = 2*pi*g^2*rho_end(epsilon)
= g^2*sqrt(4-epsilon^2). The orange exponential is only drawn for an initial
site-1 state, N >= 30, 0 < g <= 0.35, and |epsilon| < 1.8. These display thresholds
are a teaching choice, not a guarantee of an exponential regime. The comparison
neglects the resonance shift, memory effects, and finite-size reflections.
It must fail at sufficiently short and late times.

### Measurements

For an arbitrary pure initial state, the projector is Q = |psi0><psi0|.
The green curve is [P(tau)]^m * P(t-m*tau), with m = floor(t/tau).
It is the probability that all completed tests give YES and that a test at the
current time would also give YES. At exact multiples of tau, this is [P(tau)]^m.
It is not the unconditional population from a nonselective measurement channel.
For a Gaussian initial state, Q tests the whole Gaussian, not one site.
The blue bars always represent the unmeasured state.

## Validation and practical limits

Both notebooks were executed under marimo 0.25.0 with NumPy and Plotly 7.1.0.
The spin solution was compared to its analytic inversion probability and to
direct numerical integration of the time-dependent Schrodinger equation,
including positive/negative resonance, detuning, and zero field.
Chain checks cover analytic two-site dynamics, norm and energy conservation,
Gaussian normalization, return probabilities, the decoupled limit, and the
projective-measurement product and Zeno trend.

The browser animation uses a finite set of exactly computed snapshots; it does
not integrate physics frame by frame. Spin snapshots use 301 equally spaced
times. Chain snapshots combine 301 equally spaced times with extra points in
the first two time units. Playback wall-clock speed is a presentation choice;
uneven early-time spacing is visible in the time labels. Short-time zooming
uses the same figure rather than a second duplicate plot.

The notebooks have not been launched in the hosted molab service during development;
the GitHub launch links require uploading this folder first.
