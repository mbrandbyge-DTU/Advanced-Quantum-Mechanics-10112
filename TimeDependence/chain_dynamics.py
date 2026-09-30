# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=1.26", "plotly>=6.0"]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tight-binding dynamics: oscillation, decay, and “watching”
    **DTU 10112 · Time-dependent phenomena · Ballentine, pp. 332–349**

    Prepare a state, press **Play**, then **Pause**, scrub the time slider, or **Reset**.
    Changing a model control restarts at $t=0$. Site labels are **1-based** in the plots;
    NumPy indices are **0-based** in the code.

    This is the model in `time-decay-chain.nb`: one level at the end of an open chain,
    with a tunable first bond and site energy. We set $\hbar=J=1$:
    $$H=\varepsilon|1\rangle\langle1|
    -g(|1\rangle\langle2|+|2\rangle\langle1|)
    -\sum_{n=2}^{N-1}(|n\rangle\langle n+1|+|n+1\rangle\langle n|).$$
    Here $g$ is the magnitude of the first-bond hopping ($\beta=-g$ in the Mathematica
    notation). Evolution is unitary and exact for the chosen finite Hamiltonian:
    $$|\psi(t)\rangle=\sum_a e^{-iE_at}|a\rangle\langle a|\psi(0)\rangle.$$
    A long chain can mimic decay before reflections return; a finite chain is not
    an irreversible environment. The right panel measures return to the **initial state**,
    $P(t)=|\langle\psi(0)|\psi(t)\rangle|^2$, which equals site-1 occupation only when
    $|\psi(0)\rangle=|1\rangle$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Physics — the cells to inspect and change
    """)
    return


@app.cell
def _(np):
    # PHYSICS ONLY: no widgets or plotting.
    def chain_hamiltonian(N, g, epsilon):
        H = np.zeros((N, N), dtype=complex)
        for n in range(N-1):
            hopping = g if n == 0 else 1.0
            H[n, n+1] = H[n+1, n] = -hopping
        H[0, 0] = epsilon
        # STUDENT EDIT: add site energies or a final bond here.
        return H

    def initial_chain(N, kind, site, width, k):
        """site is 1-based; k is in radians per lattice spacing."""
        positions = np.arange(1, N+1)
        if kind == "Localized site":
            psi = np.zeros(N, dtype=complex)
            psi[site-1] = 1.0
        else:
            psi = np.exp(-(positions-site)**2/(4*width**2)) * np.exp(1j*k*positions)
        return psi / np.linalg.norm(psi)

    def evolve_chain(H, psi0, times):
        energies, vectors = np.linalg.eigh(H)
        coefficients = vectors.conj().T @ psi0
        phases = np.exp(-1j*np.outer(times, energies))
        states = (phases * coefficients) @ vectors.T
        # Spectral weights in the return amplitude.
        weights = np.abs(coefficients)**2
        return_probability = np.abs(phases @ weights)**2
        mean_E = np.vdot(psi0, H @ psi0).real
        variance_E = np.vdot(H @ psi0, H @ psi0).real - mean_E**2
        return states, return_probability, energies, weights, max(0.0, variance_E)

    def watched_survival(times, energies, weights, tau):
        """Joint probability of every projective test returning YES to psi0.

        After each YES the normalized state is psi0 again. Between completed
        measurements, multiply by the survival probability during the remainder.
        This is NOT the unconditional population with outcomes ignored.
        """
        def survival(t):
            return np.abs(np.exp(-1j*np.outer(np.atleast_1d(t), energies)) @ weights)**2
        completed = np.floor(times/tau + 1e-10).astype(int)
        remainder = np.maximum(0.0, times - completed*tau)
        p_step = np.clip(survival(tau)[0], 0.0, 1.0)
        return p_step**completed * survival(remainder)

    return chain_hamiltonian, evolve_chain, initial_chain, watched_survival


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Controls — prepare an experiment
    """)
    return


@app.cell
def _(mo):
    N_control = mo.ui.slider(2, 100, step=1, value=100, label="Number of sites N", show_value=True)
    g_control = mo.ui.slider(0, 1.5, step=0.05, value=0.25, label="First-bond hopping g", show_value=True)
    epsilon_control = mo.ui.slider(-4, 4, step=0.1, value=0, label="Site-1 energy epsilon", show_value=True)
    kind_control = mo.ui.dropdown(["Localized site", "Gaussian wave packet"],
                                 value="Localized site", label="Initial state")
    width_control = mo.ui.slider(0.5, 10, step=0.5, value=3, label="Gaussian width", show_value=True)
    k_control = mo.ui.slider(-1, 1, step=0.05, value=0.5, label="Gaussian k / pi", show_value=True)
    T_control = mo.ui.slider(5, 150, step=5, value=60, label="Final time", show_value=True)
    watch_control = mo.ui.checkbox(value=False, label="Show repeated YES-projective-test survival")
    tau_control = mo.ui.slider(0.05, 5, step=0.05, value=0.5, label="Measurement interval tau", show_value=True)
    mo.vstack([mo.hstack([N_control, g_control, epsilon_control], wrap=True),
               mo.hstack([kind_control, width_control, k_control], wrap=True),
               mo.hstack([T_control, watch_control, tau_control], wrap=True),
               mo.md("Width and k apply only to the Gaussian. All plotted times are in hbar/J units.")])
    return (
        N_control,
        T_control,
        epsilon_control,
        g_control,
        k_control,
        kind_control,
        tau_control,
        watch_control,
        width_control,
    )


@app.cell
def _(N_control, mo):
    # This slider is recreated if N changes, so its site index is always valid.
    site_control = mo.ui.slider(1, N_control.value, step=1, value=1,
                               label="Initial site / Gaussian centre", show_value=True)
    site_control
    return (site_control,)


@app.cell
def _(
    N_control,
    T_control,
    chain_hamiltonian,
    epsilon_control,
    evolve_chain,
    g_control,
    initial_chain,
    k_control,
    kind_control,
    np,
    site_control,
    tau_control,
    watched_survival,
    width_control,
):
    # CONNECT CONTROLS TO PHYSICS.
    N = N_control.value
    g = g_control.value
    epsilon = epsilon_control.value
    H = chain_hamiltonian(N, g, epsilon)
    psi0 = initial_chain(N, kind_control.value, site_control.value,
                         width_control.value, np.pi*k_control.value)
    # Resolve the early quadratic regime and still keep browser animation light.
    times = np.unique(np.concatenate([np.linspace(0, min(2, T_control.value), 61),
                                      np.linspace(0, T_control.value, 301)]))
    states, survival, energies, weights, energy_variance = evolve_chain(H, psi0, times)
    probabilities = np.abs(states)**2
    watched = watched_survival(times, energies, weights, tau_control.value)
    # Weak-coupling golden-rule estimate for an end site coupled to a semi-infinite
    # J=1 chain: Gamma = 2*pi*g^2*rho_end(epsilon) = g^2*sqrt(4-epsilon^2).
    show_exponential = (kind_control.value == "Localized site" and site_control.value == 1
                        and N >= 30 and 0 < g <= 0.35 and abs(epsilon) < 1.8)
    decay_rate = g**2 * np.sqrt(max(0.0, 4-epsilon**2))
    return (
        H,
        N,
        decay_rate,
        energy_variance,
        probabilities,
        psi0,
        show_exponential,
        states,
        survival,
        times,
        watched,
    )


@app.function
# VISUALIZATION ONLY: shared pattern, copied so each notebook is one file.
def add_playback(fig, times, frame_ms=70):
    def options(duration):
        return dict(mode="immediate", fromcurrent=True,
                    frame=dict(duration=duration, redraw=True),
                    transition=dict(duration=0))
    fig.update_layout(
        template="plotly_white", height=620,
        margin=dict(l=55, r=30, t=70, b=150),
        legend=dict(orientation="h", y=1.10, x=0),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.15,
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, options(frame_ms)]),
                dict(label="Pause", method="animate", args=[[None], options(0)]),
                dict(label="Reset", method="animate", args=[["0"], options(0)])])],
        sliders=[dict(active=0, x=0, y=-0.04, len=1,
            currentvalue=dict(prefix="Time t = "),
            steps=[dict(label=f"{t:.2f}", method="animate", args=[[str(i)], options(0)])
                   for i, t in enumerate(times)])])
    return fig


@app.cell
def _(
    N,
    decay_rate,
    energy_variance,
    go,
    make_subplots,
    mo,
    np,
    probabilities,
    show_exponential,
    survival,
    times,
    watch_control,
    watched,
):
    # VISUALIZATION ONLY: probability bars and return-probability curves.
    chain_fig = make_subplots(rows=1, cols=2, subplot_titles=("Site probabilities — unmeasured evolution",
                                                            "Return to the initial state"))
    chain_fig.add_trace(go.Bar(x=list(range(1, N+1)), y=probabilities[0], name="Site probability",
                              marker_color="#3182bd"), row=1, col=1)
    chain_fig.add_trace(go.Scatter(x=times, y=survival, name="Unmeasured P(t)",
                                  line=dict(color="#08519c")), row=1, col=2)
    # Only plot the quadratic approximation in its short-time domain.
    _short = energy_variance*times**2 <= 0.2
    chain_fig.add_trace(go.Scatter(x=times[_short], y=(1-energy_variance*times**2)[_short],
        name="1 - (Delta E)^2 t^2 (short time)", line=dict(color="black", dash="dot")), row=1, col=2)
    if show_exponential:
        chain_fig.add_trace(go.Scatter(x=times, y=np.exp(-decay_rate*times),
            name="Weak-coupling exp(-Gamma t)", line=dict(color="#e6550d", dash="dash")), row=1, col=2)
    if watch_control.value:
        chain_fig.add_trace(go.Scatter(x=times, y=watched,
            name="All-YES history probability", line=dict(color="#238b45")), row=1, col=2)
    _marker_index = len(chain_fig.data)
    chain_fig.add_trace(go.Scatter(x=[0], y=[1], mode="markers", showlegend=False,
                                  marker=dict(color="#08519c", size=11)), row=1, col=2)
    chain_fig.frames = [go.Frame(name=str(_i), traces=[0, _marker_index], data=[
        go.Bar(x=list(range(1, N+1)), y=probabilities[_i]),
        go.Scatter(x=[times[_i]], y=[survival[_i]])]) for _i in range(len(times))]
    chain_fig.update_xaxes(title="Site n", range=[0.5, N+0.5], row=1, col=1)
    chain_fig.update_yaxes(title="|psi_n|²", range=[0, max(0.05, 1.05*probabilities.max())], row=1, col=1)
    chain_fig.update_xaxes(title="Time t", range=[0, times[-1]], row=1, col=2)
    chain_fig.update_yaxes(title="Return / all-YES probability", range=[-0.02, 1.02], row=1, col=2)
    add_playback(chain_fig, times)
    mo.ui.plotly(chain_fig, config={"displayModeBar": False})
    return


@app.cell
def _(
    H,
    decay_rate,
    energy_variance,
    mo,
    np,
    probabilities,
    psi0,
    show_exponential,
    states,
):
    _norm_error = np.max(np.abs(probabilities.sum(axis=1)-1))
    _E0 = np.vdot(psi0, H @ psi0).real
    _Et = np.einsum('ti,ij,tj->t', states.conj(), H, states).real
    _energy_error = np.max(np.abs(_Et-_E0))
    mo.md(f"**Checks:** maximum norm error = {_norm_error:.1e}; "
          f"maximum energy drift = {_energy_error:.1e}; (Delta E)² = {energy_variance:.4f}. "
          + (f"Weak-coupling estimate Gamma = {decay_rate:.4f}. " if show_exponential else "")
          + "**The bars always show unmeasured evolution.** The optional green curve describes a separate repeated-measurement experiment.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Predict → try → explain

    1. **Two sites.** Set $N=2$, $g=1$, $\varepsilon=0$, and localize on site 1.
       Predict when site 2 becomes fully occupied. Then change $\varepsilon$:
       can transfer still be complete? Derive $P_1(t)=\cos^2(gt)$ for the resonant case.
    2. **A chain as an environment.** Compare $N=10$ and $N=100$, with $g=0.25$,
       $\varepsilon=0$, and initial site 1. Where did the missing site-1 probability go?
       Does “decay” violate norm conservation? Increase the final time to find reflections.
    3. **The first instants.** Pause near $t=0$ and zoom the right panel.
       Is the initial slope zero? Compare the dotted quadratic curve and, when shown,
       the exponential. For initial site 1, verify $(\Delta E)^2=g^2$.
    4. **Watching.** Enable the green curve, compare $\tau=1,0.5,0.1$, and keep final
       time fixed. Each ideal measurement asks “are you still in $|\psi(0)\rangle$?”:
       $$Q=|\psi(0)\rangle\langle\psi(0)|,\qquad
       P_{\rm all\ YES}(m\tau)=[P(\tau)]^m.$$
       A YES resets the normalized state to $|\psi(0)\rangle$.
       The green curve is the **joint probability of every test giving YES**, including
       survival in the interval since the most recent test. Outcomes are not averaged;
       this is not a nonselective measurement simulation. A plot refresh is not a measurement.
    5. **Wave packet.** Choose a Gaussian centred well inside a uniform long chain
       ($g=1$, $\varepsilon=0$). Predict the direction for $k=\pm\pi/2$.
       Compare with $E(k)=-2\cos k$, $v(k)=2\sin k$. Why does the packet spread?
    6. **Edit the physics.** In `chain_hamiltonian`, add `H[N//2, N//2] = 2.0`.
       Launch a packet toward it. Predict reflection/transmission, and check conservation.
       As a second task add a closing bond for a uniform ring and compare recurrences.

    **AI as a checking partner:** first expand the return amplitude through $t^2$
    yourself; then ask AI to identify the cancellation of the linear term in the
    probability. Test its result against this notebook for two different initial states.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Discussion notes — open after making predictions": mo.md(r"""
    For two resonant sites the first complete transfer is at $t=\pi/(2g)$.
    Detuning gives a maximum transfer $4g^2/(\varepsilon^2+4g^2)<1$.
    Probability spreads into the chain; total probability remains one.
    At short times, $P(t)=1-(\Delta E)^2t^2+\cdots$, whereas an exponential has
    an initial linear loss. Weak coupling can give an approximately exponential
    intermediate-time regime before finite-size reflections.
    At fixed total time $T=m\tau$, the quadratic law gives
    $[1-(\Delta E)^2\tau^2]^{T/\tau}\to1$ as $\tau\to0$.
    This limit is the ideal projective-measurement Zeno effect, not freezing by merely looking at a graph.
    """)})
    return


if __name__ == "__main__":
    app.run()
