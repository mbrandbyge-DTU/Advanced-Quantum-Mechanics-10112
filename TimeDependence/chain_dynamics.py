# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=1.26", "plotly>=6.0"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tight-binding dynamics: oscillation, decay, and "watching"
    **DTU 10112 · Time-dependent phenomena · Ballentine, pp. 332–349** · $\hbar=1$:
    $\hat H=\varepsilon_1|1\rangle\langle1|+\beta_1|1\rangle\langle2|+\beta_1|2\rangle\langle1|
    +\beta\sum_{i=2}^{N-1}(|i\rangle\langle i+1|+|i+1\rangle\langle i|)$, with $\beta=-1$.
    Try the experiment below; the physics and full write-up follow underneath it.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(controls, simulation_panel):
    simulation_panel(controls, controls.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Physics — small NumPy routines

    Site labels are **1-based** in the plots; NumPy indices are **0-based** in the
    code. These physics cells are visible and easy to edit; interface and drawing
    cells further down are collapsed by default.

    This is the model in `time-decay-chain.nb` (and the lecture slides' "Example:
    Chain"): one level at the end of an open chain, with a tunable first bond
    $\beta_1$ and site energy $\varepsilon_1$; the remaining bonds use $\beta=-1$.
    Evolution is unitary and exact for the chosen finite Hamiltonian:
    $|\psi(t)\rangle=\sum_a e^{-iE_at}|a\rangle\langle a|\psi(0)\rangle.$
    A long chain can mimic decay before reflections return; a finite chain is not
    an irreversible environment. The right panel measures return to the **initial
    state**, $P(t)=|\langle\psi(0)|\psi(t)\rangle|^2$, which equals site-1
    occupation only when $|\psi(0)\rangle=|1\rangle$.
    """)
    return


@app.cell
def _(np):
    # PHYSICS ONLY: no widgets or plotting. Add new terms here, without
    # changing any interface or drawing code.
    def chain_hamiltonian(N, beta1, epsilon1):
        """beta1 is the tunable first bond; the remaining bonds equal -1 (= beta)."""
        H = np.zeros((N, N), dtype=complex)
        for n in range(N-1):
            H[n, n+1] = H[n+1, n] = beta1 if n == 0 else -1.0
        H[0, 0] = epsilon1
        # STUDENT EDIT: add site energies or a final bond here.
        return H

    def initial_chain(N, kind, site, width, k):
        """site is 1-based; k is in radians per lattice spacing.

        Clamped to [1, N]: the site slider is rebuilt with a new upper bound
        whenever N changes, but can briefly still report a value from the
        previous, larger N (e.g. right after dragging N down) before the
        browser's new slider position reaches the kernel. Without this,
        that stale value indexes past the end of a smaller chain.
        """
        site = int(np.clip(site, 1, N))
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

    def energy_expectation(states, H):
        """<psi(t)|H|psi(t)>, as a straightforward loop over time steps."""
        return np.array([(psi.conj() @ H @ psi).real for psi in states])

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

    return (
        chain_hamiltonian,
        energy_expectation,
        evolve_chain,
        initial_chain,
        watched_survival,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interface and drawing — optional code

    The following code cells are collapsed. The complete experiment appears above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # INTERFACE ONLY. Make the controls that do not depend on each other here.
    N_control = mo.ui.slider(2, 100, step=1, value=100, label="Number of sites N", show_value=True)
    beta1_control = mo.ui.slider(-1.5, 0, step=0.05, value=-0.25, label="First-bond hopping β₁", show_value=True)
    epsilon1_control = mo.ui.slider(-4, 4, step=0.1, value=0, label="Site-1 energy ε₁", show_value=True)
    kind_control = mo.ui.dropdown(["Localized site", "Gaussian wave packet"],
                                 value="Localized site", label="Initial state")
    width_control = mo.ui.slider(0.5, 10, step=0.5, value=3, label="Gaussian width", show_value=True)
    k_control = mo.ui.slider(-1, 1, step=0.05, value=0.5, label="Gaussian k / pi", show_value=True)
    T_control = mo.ui.slider(5, 150, step=5, value=60, label="Final time", show_value=True)
    watch_control = mo.ui.checkbox(value=False, label="Show repeated YES-projective-test survival")
    tau_control = mo.ui.slider(0.05, 5, step=0.05, value=0.5, label="Measurement interval tau", show_value=True)
    speed_control = mo.ui.slider(0.25, 4, step=0.25, value=1, label="Playback speed", show_value=True)
    return (
        N_control,
        T_control,
        beta1_control,
        epsilon1_control,
        k_control,
        kind_control,
        speed_control,
        tau_control,
        watch_control,
        width_control,
    )


@app.cell(hide_code=True)
def _(N_control, mo):
    # INTERFACE ONLY. This slider is recreated whenever N changes, so its
    # site index is always valid.
    site_control = mo.ui.slider(1, N_control.value, step=1, value=1,
                               label="Initial site / Gaussian centre", show_value=True)
    return (site_control,)


@app.cell(hide_code=True)
def _(
    N_control,
    T_control,
    beta1_control,
    epsilon1_control,
    k_control,
    kind_control,
    mo,
    site_control,
    speed_control,
    tau_control,
    watch_control,
    width_control,
):
    # INTERFACE ONLY. Bundle every control into one reactive dictionary, the
    # same pattern used in the spin notebook: reading controls.value below is
    # what makes the panel re-render when any single control changes.
    controls = mo.ui.dictionary({
        "N": N_control, "beta1": beta1_control, "epsilon1": epsilon1_control,
        "kind": kind_control, "width": width_control, "k": k_control,
        "T": T_control, "watch": watch_control, "tau": tau_control,
        "speed": speed_control, "site": site_control,
    })
    return (controls,)


@app.cell(hide_code=True)
def _(go, make_subplots, np):
    # DRAWING ONLY: probability bars and return-probability curves.
    def add_playback(fig, times, frame_ms=70):
        def options(duration):
            return dict(mode="immediate", fromcurrent=True,
                        frame=dict(duration=duration, redraw=True),
                        transition=dict(duration=0))
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.22,
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, options(frame_ms)]),
                    dict(label="Pause", method="animate", args=[[None], options(0)]),
                    dict(label="Reset", method="animate", args=[["0"], options(0)])])],
            sliders=[dict(active=0, x=0, y=-0.07, len=1, font=dict(size=9),
                currentvalue=dict(prefix="Time t = ", font=dict(size=11)),
                steps=[dict(label=f"{t:.2f}", method="animate", args=[[str(i)], options(0)])
                       for i, t in enumerate(times)])])
        return fig

    def chain_figure(N, times, probabilities, survival, energy_variance,
                     decay_rate, show_exponential, watch, watched, frame_ms):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Site probabilities (unmeasured)", "Return to initial state"))
        fig.add_trace(go.Bar(x=list(range(1, N+1)), y=probabilities[0], name="Site probability",
                             marker_color="#3182bd"), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=survival, name="P(t)",
                                 line=dict(color="#08519c")), row=1, col=2)
        # Only plot the quadratic approximation in its short-time domain.
        short = energy_variance*times**2 <= 0.2
        fig.add_trace(go.Scatter(x=times[short], y=(1-energy_variance*times**2)[short],
            name="Short-time quadratic", line=dict(color="black", dash="dot")), row=1, col=2)
        if show_exponential:
            fig.add_trace(go.Scatter(x=times, y=np.exp(-decay_rate*times),
                name="Weak-coupling exp(−Γt)", line=dict(color="#e6550d", dash="dash")), row=1, col=2)
        if watch:
            fig.add_trace(go.Scatter(x=times, y=watched,
                name="All-YES survival", line=dict(color="#238b45")), row=1, col=2)
        marker_index = len(fig.data)
        fig.add_trace(go.Scatter(x=[0], y=[1], mode="markers", showlegend=False,
                                 marker=dict(color="#08519c", size=9)), row=1, col=2)
        fig.frames = [go.Frame(name=str(i), traces=[0, marker_index], data=[
            go.Bar(x=list(range(1, N+1)), y=probabilities[i]),
            go.Scatter(x=[times[i]], y=[survival[i]])]) for i in range(len(times))]
        fig.update_xaxes(title="Site n", range=[0.5, N+0.5], row=1, col=1)
        fig.update_yaxes(title="|ψₙ|²", range=[0, max(0.05, 1.05*probabilities.max())], row=1, col=1)
        fig.update_xaxes(title="Time t", range=[0, times[-1]], row=1, col=2)
        fig.update_yaxes(title="P(t)", range=[-0.02, 1.02], row=1, col=2)
        # The legend has up to 5 entries and wraps onto 2 lines at this width;
        # push it well above the subplot titles (rather than just above the
        # plot area) so the wrapped lines never overlap the titles below them.
        fig.update_layout(template="plotly_white", height=330,
            font=dict(size=11),
            margin=dict(l=40, r=10, t=78, b=78),
            legend=dict(orientation="h", y=1.42, x=0, font=dict(size=10)))
        add_playback(fig, times, frame_ms=frame_ms)
        return fig

    return (chain_figure,)


@app.cell(hide_code=True)
def _(
    chain_figure,
    chain_hamiltonian,
    energy_expectation,
    evolve_chain,
    initial_chain,
    mo,
    np,
    watched_survival,
):
    # INTERFACE ADAPTER: read controls, call the physics, assemble ONE output.
    def simulation_panel(controls, values):
        N, beta1, epsilon1 = values["N"], values["beta1"], values["epsilon1"]
        H = chain_hamiltonian(N, beta1, epsilon1)
        psi0 = initial_chain(N, values["kind"], values["site"], values["width"], np.pi*values["k"])
        # Resolve the early quadratic regime and still keep browser animation light.
        times = np.unique(np.concatenate([np.linspace(0, min(2, values["T"]), 61),
                                          np.linspace(0, values["T"], 301)]))
        states, survival, energies, weights, energy_variance = evolve_chain(H, psi0, times)
        probabilities = np.abs(states)**2
        watched = watched_survival(times, energies, weights, values["tau"])
        # Weak-coupling golden-rule estimate for an end site coupled to a semi-infinite
        # chain (remaining bonds = -1): Gamma = 2*pi*beta1^2*rho_end(epsilon1) = beta1^2*sqrt(4-epsilon1^2).
        show_exponential = (values["kind"] == "Localized site" and values["site"] == 1
                            and N >= 30 and -0.35 <= beta1 < 0 and abs(epsilon1) < 1.8)
        decay_rate = beta1**2 * np.sqrt(max(0.0, 4-epsilon1**2))
        frame_ms = max(5, round(70 / values["speed"]))

        fig = chain_figure(N, times, probabilities, survival, energy_variance,
                           decay_rate, show_exponential, values["watch"], watched, frame_ms)
        chart = mo.ui.plotly(fig, config={"displayModeBar": False})

        norm_error = np.max(np.abs(probabilities.sum(axis=1)-1))
        E0 = np.vdot(psi0, H @ psi0).real
        Et = energy_expectation(states, H)
        energy_error = np.max(np.abs(Et-E0))
        details = (f"Norm error {norm_error:.1e}; energy drift {energy_error:.1e}; "
                   f"(ΔE)² = {energy_variance:.4f}. "
                   + (f"Weak-coupling Γ = {decay_rate:.4f}. " if show_exponential else "")
                   + "Bars always show unmeasured evolution; the green curve is a separate "
                     "repeated-measurement experiment.")

        sidebar = mo.vstack([
            mo.hstack([controls["N"], controls["beta1"], controls["epsilon1"]], wrap=True, gap=0.6),
            mo.hstack([controls["kind"], controls["width"], controls["k"]], wrap=True, gap=0.6),
            mo.hstack([controls["T"], controls["watch"], controls["tau"]], wrap=True, gap=0.6),
            mo.hstack([controls["site"], controls["speed"]], wrap=True, gap=0.6),
            mo.md("Width and k apply only to the Gaussian; times are in ħ units."),
        ], gap=0.4)

        panel = mo.vstack([
            mo.md("### Chain experiment"),
            sidebar,
            chart,
            mo.md(details),
        ], gap=0.3).style({"border": "1px solid #dce3eb", "border-radius": "12px",
                           "padding": "12px", "margin-bottom": "14px"})
        return panel

    return (simulation_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict → try → explain

    1. **Two sites.** Set $N=2$, $\beta_1=-1$, $\varepsilon_1=0$, and localize on site 1.
       Predict when site 2 becomes fully occupied. Then change $\varepsilon_1$:
       can transfer still be complete? Derive $P_1(t)=\cos^2(\beta_1 t)$ for the resonant case.
    2. **A chain as an environment.** Compare $N=10$ and $N=100$, with $\beta_1=-0.25$,
       $\varepsilon_1=0$, and initial site 1. Where did the missing site-1 probability go?
       Does "decay" violate norm conservation? Increase the final time to find reflections.
    3. **The first instants.** Pause near $t=0$ and zoom the right panel.
       Is the initial slope zero? Compare the dotted quadratic curve and, when shown,
       the exponential. For initial site 1, verify $(\Delta E)^2=\beta_1^2$.
    4. **Watching.** Enable the green curve, compare $\tau=1,0.5,0.1$, and keep final
       time fixed. Each ideal measurement asks "are you still in $|\psi(0)\rangle$?":
       $Q=|\psi(0)\rangle\langle\psi(0)|,\qquad
       P_{\rm all\ YES}(m\tau)=[P(\tau)]^m.$
       A YES resets the normalized state to $|\psi(0)\rangle$.
       The green curve is the **joint probability of every test giving YES**, including
       survival in the interval since the most recent test. Outcomes are not averaged;
       this is not a nonselective measurement simulation. A plot refresh is not a measurement.
    5. **Wave packet.** Choose a Gaussian centred well inside a uniform long chain
       ($\beta_1=-1$, $\varepsilon_1=0$). Predict the direction for $k=\pm\pi/2$.
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
    For two resonant sites the first complete transfer is at $t=\pi/(2|\beta_1|)$.
    Detuning gives a maximum transfer $4\beta_1^2/(\varepsilon_1^2+4\beta_1^2)<1$.
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
