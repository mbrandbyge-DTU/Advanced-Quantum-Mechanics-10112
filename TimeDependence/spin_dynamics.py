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
    # Spin-1/2: precession and resonance
    **DTU 10112 · Time-dependent phenomena · Ballentine, pp. 332–349**

    Choose an initial state and a Hamiltonian, then press **Play** below the plot.
    **Pause** and drag the time slider to examine any instant; **Reset** returns to $t=0$.
    Changing a model control prepares a new experiment at $t=0$.

    We use your lecture's sign convention, $\omega_i=\gamma B_i$, and set $\hbar=1$:
    $$H(t)=-\frac12[\omega_0\sigma_z+
    \omega_1\cos(\omega t)\sigma_x+\omega_1\sin(\omega t)\sigma_y].$$
    The rotating field is **circularly polarized**, so the rotating-frame solution is exact:
    $$R(t)=e^{-i\omega t\sigma_z/2},\qquad
    H_{\rm eff}=-\frac12[(\omega_0+\omega)\sigma_z+\omega_1\sigma_x],\qquad
    |\psi(t)\rangle=R(t)e^{-iH_{\rm eff}t}|\psi(0)\rangle.$$
    Resonance is $\omega=-\omega_0$ in this convention. Frequencies are angular
    frequencies in one arbitrary inverse-time unit; no relaxation or ensemble averaging is included.
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
    # PHYSICS ONLY: no plotting or widgets.
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.diag([1, -1]).astype(complex)

    def initial_spin(theta, phi):
        """Angles in radians; basis is |+z>, |-z>."""
        return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])

    def evolve_spin(psi0, times, omega0, omega1, omega):
        """Exact solution for a static z field plus a circular rotating field."""
        H_eff = -0.5 * ((omega0 + omega) * sz + omega1 * sx)
        energies, vectors = np.linalg.eigh(H_eff)
        coefficients = vectors.conj().T @ psi0
        # One row per time; columns are spin components.
        psi_rot = (np.exp(-1j*np.outer(times, energies)) * coefficients) @ vectors.T
        rotation = np.column_stack([np.exp(-0.5j*omega*times),
                                    np.exp(+0.5j*omega*times)])
        psi_lab = rotation * psi_rot
        return psi_lab, psi_rot

    def bloch_vector(states):
        """r = <sigma>; <S> = hbar*r/2, and pure states have |r|=1."""
        return np.column_stack([
            np.einsum('ti,ij,tj->t', states.conj(), sigma, states).real
            for sigma in (sx, sy, sz)])

    return bloch_vector, evolve_spin, initial_spin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Controls — prepare an experiment
    """)
    return


@app.cell
def _(mo):
    mode = mo.ui.dropdown(["Static field (precession)", "Rotating field (resonance)"],
                          value="Static field (precession)", label="Experiment")
    initial = mo.ui.dropdown(["+z", "-z", "+x", "+y", "Custom angles"],
                             value="+x", label="Initial spin")
    theta = mo.ui.slider(0, 180, step=5, value=60, label="Custom theta (degrees)", show_value=True)
    phi = mo.ui.slider(-180, 180, step=5, value=0, label="Custom phi (degrees)", show_value=True)
    omega0 = mo.ui.slider(-2, 2, step=0.1, value=1, label="omega_0", show_value=True)
    omega1 = mo.ui.slider(0, 1, step=0.05, value=0.3, label="omega_1 (rotating field)", show_value=True)
    omega = mo.ui.slider(-3, 3, step=0.1, value=-1, label="omega (signed drive frequency)", show_value=True)
    frame = mo.ui.dropdown(["Laboratory", "Rotating"], value="Laboratory", label="Bloch-sphere frame")
    end_time = mo.ui.slider(5, 60, step=5, value=25, label="Final time", show_value=True)
    mo.vstack([mo.hstack([mode, initial, frame], wrap=True),
               mo.hstack([theta, phi], wrap=True),
               mo.hstack([omega0, omega1, omega], wrap=True), end_time,
               mo.md("Custom angles apply only to **Custom angles**. Static mode ignores omega_1 and omega; its two frame views coincide.")])
    return end_time, frame, initial, mode, omega, omega0, omega1, phi, theta


@app.cell
def _(
    bloch_vector,
    end_time,
    evolve_spin,
    frame,
    initial,
    initial_spin,
    mode,
    np,
    omega,
    omega0,
    omega1,
    phi,
    theta,
):
    # CONNECT CONTROLS TO PHYSICS; visualization is in the next cell.
    _angles = {"+z": (0, 0), "-z": (np.pi, 0), "+x": (np.pi/2, 0), "+y": (np.pi/2, np.pi/2)}
    _theta, _phi = _angles.get(initial.value, (np.deg2rad(theta.value), np.deg2rad(phi.value)))
    spin0 = initial_spin(_theta, _phi)
    spin_times = np.linspace(0, end_time.value, 301)
    w0 = omega0.value
    w1 = omega1.value if mode.value == "Rotating field (resonance)" else 0.0
    w = omega.value if mode.value == "Rotating field (resonance)" else 0.0
    lab_states, rotating_states = evolve_spin(spin0, spin_times, w0, w1, w)
    spin_bloch = bloch_vector(lab_states if frame.value == "Laboratory" else rotating_states)
    p_down = np.abs(lab_states[:, 1])**2
    if frame.value == "Laboratory":
        frequency_vectors = np.column_stack([w1*np.cos(w*spin_times), w1*np.sin(w*spin_times),
                                            np.full_like(spin_times, w0)])
    else:
        frequency_vectors = np.tile([w1, 0, w0+w], (len(spin_times), 1))
    # Normalize arrows for display only; retain physical magnitudes in w0, w1, w.
    _frequency_norm = np.linalg.norm(frequency_vectors, axis=1, keepdims=True)
    frequency_vectors = np.divide(frequency_vectors, _frequency_norm,
                                  out=np.zeros_like(frequency_vectors), where=_frequency_norm>0)
    return (
        frequency_vectors,
        lab_states,
        p_down,
        spin_bloch,
        spin_times,
        w,
        w0,
        w1,
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
    frame,
    frequency_vectors,
    go,
    make_subplots,
    mo,
    np,
    p_down,
    spin_bloch,
    spin_times,
):
    # VISUALIZATION ONLY: build a sphere and update selected traces per frame.
    spin_fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                            subplot_titles=(f"Bloch sphere — {frame.value.lower()} frame", "Probability to measure -z"))
    # Three great circles give a light, transparent sphere.
    _a = np.linspace(0, 2*np.pi, 100)
    for _xyz in [(np.cos(_a), np.sin(_a), np.zeros_like(_a)),
                 (np.cos(_a), np.zeros_like(_a), np.sin(_a)),
                 (np.zeros_like(_a), np.cos(_a), np.sin(_a))]:
        spin_fig.add_trace(go.Scatter3d(x=_xyz[0], y=_xyz[1], z=_xyz[2], mode="lines",
            line=dict(color="#cccccc", width=2), showlegend=False, hoverinfo="skip"), row=1, col=1)

    def spin_frame_traces(i):
        r = spin_bloch[i]
        b = frequency_vectors[i]
        return [
            go.Scatter3d(x=spin_bloch[:i+1, 0], y=spin_bloch[:i+1, 1], z=spin_bloch[:i+1, 2],
                         mode="lines", line=dict(color="#3182bd", width=3), name="Spin trajectory"),
            go.Scatter3d(x=[0, r[0]], y=[0, r[1]], z=[0, r[2]], mode="lines+markers",
                         line=dict(color="#08519c", width=6), marker=dict(size=[0, 5]), name="<sigma>"),
            go.Scatter3d(x=[0, b[0]], y=[0, b[1]], z=[0, b[2]], mode="lines",
                         line=dict(color="#238b45", width=5), name="Frequency axis (unit length)"),
            go.Scatter(x=[spin_times[i]], y=[p_down[i]], mode="markers",
                       marker=dict(color="#08519c", size=11), name="Current time", showlegend=False)]

    _dynamic_indices = [3, 4, 5, 7]
    for _tr in spin_frame_traces(0)[:3]:
        spin_fig.add_trace(_tr, row=1, col=1)
    spin_fig.add_trace(go.Scatter(x=spin_times, y=p_down, mode="lines", name="P(-z)",
                                 line=dict(color="#08519c")), row=1, col=2)
    spin_fig.add_trace(spin_frame_traces(0)[3], row=1, col=2)
    spin_fig.frames = [go.Frame(name=str(_i), data=spin_frame_traces(_i), traces=_dynamic_indices)
                       for _i in range(len(spin_times))]
    spin_fig.update_layout(scene=dict(
        xaxis=dict(title="<sigma_x>", range=[-1.1, 1.1]),
        yaxis=dict(title="<sigma_y>", range=[-1.1, 1.1]),
        zaxis=dict(title="<sigma_z>", range=[-1.1, 1.1]), aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))))
    spin_fig.update_xaxes(title="Time t", range=[0, spin_times[-1]], row=1, col=2)
    spin_fig.update_yaxes(title="P(-z)", range=[-0.02, 1.02], row=1, col=2)
    add_playback(spin_fig, spin_times)
    mo.ui.plotly(spin_fig, config={"displayModeBar": False})
    return


@app.cell
def _(lab_states, mo, np, w, w0, w1):
    _norm_error = np.max(np.abs(np.sum(np.abs(lab_states)**2, axis=1)-1))
    _detuning = w0 + w
    _rabi = np.hypot(_detuning, w1)
    mo.md(f"**Numerical check:** maximum norm error = {_norm_error:.1e}. "
          f"**Rotating-field parameters:** detuning = {_detuning:.2f}, "
          f"generalized Rabi frequency = {_rabi:.3f}. "
          + (f"On-resonance pi-pulse time = {np.pi/w1:.3f}." if w1 > 0 else "Transverse drive is off."))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Predict → try → explain

    1. **Stationary state versus stationary spin direction.** In static mode compare
       $|+z\rangle$ and $|+x\rangle$. Predict which probabilities change and which do not.
       Does a constant Bloch vector imply the ket has no time dependence?
    2. **Find resonance.** Select rotating field, initial $|+z\rangle$, $\omega_0=1$,
       $\omega_1=0.3$. Find the signed $\omega$ that gives a complete spin flip.
       Switch frames: why is one trajectory simpler? Predict the first flip time.
    3. **Detune.** Change $\omega$ by $0.3$. Predict whether oscillations become
       faster or slower, and whether their amplitude increases or decreases.
    4. **Edit the physics.** Change the sign of `w0` through its slider and find resonance
       again. Then add an `initial_spin` preset for $|-x\rangle$ in the
       control-to-physics cell. Check its initial Bloch vector before playing.
    5. **Extension (a different model).** Replace the circular field by a linearly
       polarized field, $H=-[\omega_0\sigma_z+\omega_1\cos(\omega t)\sigma_x]/2$.
       The constant `H_eff` used here no longer solves that problem exactly.
       Write a time-stepping solver and check norm and step-size convergence.
       Compare its weak-drive resonance with the circular-field case.

    **AI as a checking partner:** derive the rotating-frame Hamiltonian yourself, then ask:
    “Check each sign in my derivation using this $H(t)$ and $R(t)$; do not replace my
    convention.” Test its answer by changing the sign of $\omega_0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Discussion notes — open after making predictions": mo.md(r"""
    For a static field, $|+z\rangle$ gains a global phase, while the relative phase
    of $|+x\rangle$ evolves. Its $z$ probabilities stay $1/2$, but its Bloch vector precesses.
    For initial $|+z\rangle$ and $\alpha=\sqrt{(\omega_0+\omega)^2+\omega_1^2}$,
    $$P_{-z}(t)=\frac{\omega_1^2}{\alpha^2}\sin^2(\alpha t/2).$$
    At resonance the first complete flip is at $t=\pi/|\omega_1|$.
    Detuning increases $\alpha$ while reducing the maximum flip probability.
    If both detuning and drive vanish, take the continuous limit $P_{-z}=0$.
    """)})
    return


if __name__ == "__main__":
    app.run()
