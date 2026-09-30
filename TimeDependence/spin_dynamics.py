# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=1.26", "plotly>=6.0"]
# ///

import marimo

__generated_with = "0.25.0"
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
    # Spin-1/2 dynamics
    **DTU 10112 · Time-dependent phenomena · Ballentine, pp. 332–349**

    The two experiments below each keep **controls, plots, and playback together**.
    The short physics cells are visible. Interface and drawing cells are collapsed
    by default; open them only if you want to change the presentation.

    We use the slides' sign convention, $\omega_i=\gamma B_i$, and $\hbar=1$:
    $$H(t)=-\frac12[\omega_0\sigma_z+\omega_1\cos(\omega t)\sigma_x
    +\omega_1\sin(\omega t)\sigma_y].$$
    For the circular drive the exact rotating-frame solution is
    $$H_{\rm eff}=-\tfrac12[(\omega_0+\omega)\sigma_z+\omega_1\sigma_x],
    \qquad |\psi(t)\rangle=e^{-i\omega t\sigma_z/2}e^{-iH_{\rm eff}t}|\psi(0)\rangle.$$
    Resonance is $\omega=-\omega_0$. Frequencies are angular frequencies in arbitrary
    inverse-time units; this pure-state model includes no relaxation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Physics — small NumPy routines
    """)
    return


@app.cell
def _(np):
    # Basis: |+z>, |-z>. All angles in radians.
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    # Add new student presets here, without changing any plotting code.
    spin_presets = {
        "+z": (0, 0),
        "-z": (np.pi, 0),
        "+x": (np.pi/2, 0),
        "+y": (np.pi/2, np.pi/2),
    }

    def initial_spin(theta, phi):
        return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])

    return initial_spin, spin_presets, sx, sy, sz


@app.cell
def _(np, sx, sz):
    def evolve_spin(psi0, times, omega0, omega1, omega):
        """Exact circular-drive solution; static precession: omega1 = omega = 0."""
        H_eff = -0.5 * ((omega0 + omega)*sz + omega1*sx)
        energies, eigenvectors = np.linalg.eigh(H_eff)
        coefficients = eigenvectors.conj().T @ psi0

        # For each time: evolve eigenstate coefficients, then rotate back.
        psi_rot = np.zeros((len(times), 2), dtype=complex)
        psi_lab = np.zeros((len(times), 2), dtype=complex)
        for i, t in enumerate(times):
            phases = np.exp(-1j*energies*t)
            psi_rot[i] = eigenvectors @ (phases*coefficients)
            R = np.diag([np.exp(-1j*omega*t/2), np.exp(+1j*omega*t/2)])
            psi_lab[i] = R @ psi_rot[i]
        return psi_lab, psi_rot

    return (evolve_spin,)


@app.cell
def _(np, sx, sy, sz):
    def bloch_vector(states):
        """r = <sigma>; the spin expectation is r/2 when hbar = 1."""
        r = np.zeros((len(states), 3))
        for i, psi in enumerate(states):
            r[i, 0] = (psi.conj() @ sx @ psi).real
            r[i, 1] = (psi.conj() @ sy @ psi).real
            r[i, 2] = (psi.conj() @ sz @ psi).real
        return r

    return (bloch_vector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interface and drawing — optional code

    The following code cells are collapsed. The complete experiments appear below them.
    """)
    return


@app.cell(hide_code=True)
def _(mo, spin_presets):
    # INTERFACE ONLY. Make widgets here, but display them together with the figure.
    def make_controls(rotating):
        controls = {
            "initial": mo.ui.dropdown(list(spin_presets) + ["Custom angles"],
                                      value="+z" if rotating else "+x", label="Initial spin"),
            "theta": mo.ui.slider(0, 180, step=5, value=60, label="Custom θ (degrees)", show_value=True),
            "phi": mo.ui.slider(-180, 180, step=5, value=0, label="Custom φ (degrees)", show_value=True),
            "omega0": mo.ui.slider(-2, 2, step=0.1, value=1, label="ω₀", show_value=True),
            "end": mo.ui.slider(5, 60, step=5, value=25, label="Final time", show_value=True),
        }
        if rotating:
            controls.update({
                "omega1": mo.ui.slider(0, 1, step=0.05, value=0.3, label="ω₁ (drive strength)", show_value=True),
                "omega": mo.ui.slider(-3, 3, step=0.1, value=-1, label="ω (signed drive frequency)", show_value=True),
                "frame": mo.ui.dropdown(["Laboratory", "Rotating"], value="Laboratory", label="View frame"),
            })
        # A reactive dictionary binds every child control to this UI element.
        return mo.ui.dictionary(controls)

    return (make_controls,)


@app.cell(hide_code=True)
def _(go, make_subplots, np):
    # DRAWING ONLY: a solid tube and cone, with the arrow tip exactly at the vector.
    def arrow3d(vector, color, name):
        vector = np.asarray(vector, dtype=float)
        length = np.linalg.norm(vector)
        if length < 1e-12:
            return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], color=color,
                             name=name, showlegend=True, hoverinfo="skip")
        direction = vector/length
        reference = np.array([0., 0., 1.]) if abs(direction[2]) < 0.9 else np.array([0., 1., 0.])
        e1 = np.cross(direction, reference)
        e1 = e1/np.linalg.norm(e1)
        e2 = np.cross(direction, e1)
        n = 20
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        circle = np.cos(angles)[:, None]*e1 + np.sin(angles)[:, None]*e2
        head_length = min(0.20, 0.25*length)
        neck = vector - head_length*direction
        vertices = np.vstack([0.020*circle, neck+0.020*circle,
                              neck+0.075*circle, vector, [0., 0., 0.], neck])
        faces = []
        tip, base, centre = 3*n, 3*n+1, 3*n+2
        for j in range(n):
            k = (j+1) % n
            faces.extend([(j, k, n+j), (k, n+k, n+j),       # tube
                          (base, k, j),                    # tube bottom
                          (n+j, n+k, 2*n+j), (n+k, 2*n+k, 2*n+j),
                          (2*n+j, 2*n+k, tip),             # cone
                          (centre, 2*n+k, 2*n+j)])         # cone bottom
        faces = np.array(faces)
        return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color,
                         name=name, showlegend=True, hoverinfo="skip", flatshading=False,
                         lighting=dict(ambient=0.45, diffuse=0.8, specular=0.4, roughness=0.35),
                         lightposition=dict(x=100, y=100, z=200))

    def spin_figure(times, r, frequency_axis, p_down, p_x, view):
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                            column_widths=[0.53, 0.47], horizontal_spacing=0.07,
                            subplot_titles=(f"Bloch sphere · {view.lower()}", "Measurement probabilities"))
        a = np.linspace(0, 2*np.pi, 70)
        b = np.linspace(0, np.pi, 35)
        sphere = go.Surface(x=np.outer(np.cos(a), np.sin(b)), y=np.outer(np.sin(a), np.sin(b)),
                            z=np.outer(np.ones_like(a), np.cos(b)), opacity=0.12,
                            colorscale=[[0, "#a8c4dc"], [1, "#a8c4dc"]], showscale=False,
                            hoverinfo="skip", name="Sphere")
        fig.add_trace(sphere, row=1, col=1)
        for xyz in [(np.cos(a), np.sin(a), np.zeros_like(a)),
                    (np.cos(a), np.zeros_like(a), np.sin(a)),
                    (np.zeros_like(a), np.cos(a), np.sin(a))]:
            fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode="lines",
                          line=dict(color="#a5b3bf", width=2), showlegend=False, hoverinfo="skip"), row=1, col=1)
        for axis, label in zip(np.eye(3), ["x", "y", "z"]):
            endpoints = np.array([-1.15*axis, 1.18*axis])
            fig.add_trace(go.Scatter3d(x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2],
                          mode="lines+text", text=["", label], textposition="top center",
                          line=dict(color="#555555", width=2), showlegend=False, hoverinfo="skip"), row=1, col=1)

        def traces(i):
            return [
                go.Scatter3d(x=r[:i+1, 0], y=r[:i+1, 1], z=r[:i+1, 2], mode="lines",
                             line=dict(color="#4a7ac0", width=4), showlegend=False, hoverinfo="skip"),
                arrow3d(r[i], "#235ba8", "Spin ⟨σ⟩"),
                arrow3d(frequency_axis[i], "#dd8b22", "Frequency axis"),
                go.Scatter(x=[times[i], times[i]], y=[p_down[i], p_x[i]], mode="markers",
                           marker=dict(size=9, color=["#235ba8", "#b04d86"]), showlegend=False),
            ]
        dynamic = [len(fig.data), len(fig.data)+1, len(fig.data)+2]
        for trace in traces(0)[:3]:
            fig.add_trace(trace, row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=p_down, name="P(−z)",
                                 line=dict(color="#235ba8", width=2.5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=times, y=p_x, name="P(+x), lab",
                                 line=dict(color="#b04d86", width=2, dash="dot")), row=1, col=2)
        dynamic.append(len(fig.data))
        fig.add_trace(traces(0)[3], row=1, col=2)
        fig.frames = [go.Frame(name=str(i), data=traces(i), traces=dynamic) for i in range(len(times))]
        axis_style = dict(visible=False, range=[-1.3, 1.3])
        fig.update_layout(scene=dict(xaxis=axis_style, yaxis=axis_style, zaxis=axis_style,
                                    aspectmode="cube", bgcolor="white",
                                    camera=dict(eye=dict(x=0.90, y=0.90, z=0.65))))
        fig.update_xaxes(title="Time t", range=[0, times[-1]], showgrid=True, gridcolor="#edf0f3")
        fig.update_yaxes(title="Probability", range=[-0.03, 1.03], tickvals=[0, 0.5, 1], gridcolor="#edf0f3")

        def animation_options(duration):
            return dict(mode="immediate", fromcurrent=True,
                        frame=dict(duration=duration, redraw=True), transition=dict(duration=0))
        fig.update_layout(template="plotly_white", height=510,
            font=dict(family="Georgia, serif", size=13, color="#243247"),
            margin=dict(l=45, r=15, t=65, b=110),
            legend=dict(orientation="h", x=0, y=1.14, font=dict(size=12)),
            updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.18,
                buttons=[dict(label="▶ Play", method="animate", args=[None, animation_options(65)]),
                         dict(label="Pause", method="animate", args=[[None], animation_options(0)]),
                         dict(label="Reset", method="animate", args=[["0"], animation_options(0)])])],
            sliders=[dict(active=0, x=0, y=-0.04, len=1, ticklen=0, font=dict(size=10),
                currentvalue=dict(prefix="t = ", font=dict(size=13)),
                steps=[dict(label=f"{t:.2f}", method="animate", args=[[str(i)], animation_options(0)])
                       for i, t in enumerate(times)])])
        return fig

    return (spin_figure,)


@app.cell(hide_code=True)
def _(
    bloch_vector,
    evolve_spin,
    initial_spin,
    mo,
    np,
    spin_figure,
    spin_presets,
):
    # INTERFACE ADAPTER: read controls, call the physics, assemble ONE output.
    def simulation_panel(controls, values, rotating):
        initial = values["initial"]
        if initial == "Custom angles":
            theta = np.deg2rad(values["theta"])
            phi = np.deg2rad(values["phi"])
        else:
            theta, phi = spin_presets[initial]
        psi0 = initial_spin(theta, phi)
        times = np.linspace(0, values["end"], 201)
        w0 = values["omega0"]
        w1 = values["omega1"] if rotating else 0.0
        w = values["omega"] if rotating else 0.0
        view = values["frame"] if rotating else "Laboratory"
        lab, rot = evolve_spin(psi0, times, w0, w1, w)
        r_lab = bloch_vector(lab)
        r = bloch_vector(rot) if view == "Rotating" else r_lab
        p_down = np.abs(lab[:, 1])**2
        p_x = (1+r_lab[:, 0])/2
        frequency_axis = np.zeros((len(times), 3))
        for i, t in enumerate(times):
            v = np.array([w1, 0, w0+w]) if view == "Rotating" else np.array([w1*np.cos(w*t), w1*np.sin(w*t), w0])
            if np.linalg.norm(v) > 0:
                frequency_axis[i] = v/np.linalg.norm(v)
        fig = spin_figure(times, r, frequency_axis, p_down, p_x, view)
        chart = mo.ui.plotly(fig, config={"displayModeBar": False, "responsive": True})
        # No widget has a separate output cell; controls sit inside this card.
        widgets = [controls["initial"], controls["omega0"]]
        if rotating:
            widgets += [controls["omega1"], controls["omega"], controls["frame"]]
        widgets += [controls["end"], mo.accordion({"Custom initial angles":
                   mo.vstack([controls["theta"], controls["phi"],
                              mo.md("Used only with **Custom angles**.")])})]
        sidebar = mo.vstack(widgets, gap=0.6)
        norm_error = max(abs(np.sum(abs(psi)**2)-1) for psi in lab)
        details = f"Norm error: {norm_error:.1e}. "
        if rotating:
            details += f"Detuning ω₀+ω = {w0+w:.2f}; Rabi frequency = {np.sqrt((w0+w)**2+w1**2):.3f}. "
            if w1 > 0:
                details += f"On-resonance π-pulse time: {np.pi/w1:.3f}."
        title = "2. Magnetic resonance" if rotating else "1. Precession in a static field"
        hint = ("Start at +z; set ω = −ω₀ for full inversion. Switch to the rotating frame."
                if rotating else "Compare +x and +z; watch P(+x) as well as the spin direction.")
        panel = mo.vstack([
            mo.md(f"### {title}\n{hint}"),
            mo.hstack([sidebar, chart], widths=[1, 4], align="start", gap=1.0),
            mo.md("**Play · Pause · Reset · time slider** below the plots. Changes restart at t = 0. "
                  "Drag the sphere to rotate the view. The orange arrow is a unit frequency axis.\n\n"+details),
        ], gap=0.7).style({"border":"1px solid #dce3eb", "border-radius":"12px", "padding":"18px", "margin-bottom":"20px"})
        return panel, fig

    return (simulation_panel,)


@app.cell(hide_code=True)
def _(make_controls):
    precession_controls = make_controls(rotating=False)
    return (precession_controls,)


@app.cell(hide_code=True)
def _(precession_controls, simulation_panel):
    precession_panel, precession_fig = simulation_panel(precession_controls, precession_controls.value, rotating=False)
    precession_panel
    return


@app.cell(hide_code=True)
def _(make_controls):
    resonance_controls = make_controls(rotating=True)
    return (resonance_controls,)


@app.cell(hide_code=True)
def _(resonance_controls, simulation_panel):
    resonance_panel, resonance_fig = simulation_panel(resonance_controls, resonance_controls.value, rotating=True)
    resonance_panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict → try → explain

    1. **Stationary state versus stationary spin direction.** In the precession panel compare
       $|+z\rangle$ and $|+x\rangle$. Predict which probabilities change and which do not.
       Does a constant Bloch vector imply the ket has no time dependence?
    2. **Find resonance.** In the resonance panel, select initial $|+z\rangle$, $\omega_0=1$,
       $\omega_1=0.3$. Find the signed $\omega$ that gives a complete spin flip.
       Switch frames: why is one trajectory simpler? Predict the first flip time.
    3. **Detune.** Change $\omega$ by $0.3$. Predict whether oscillations become
       faster or slower, and whether their amplitude increases or decreases.
    4. **Edit the physics.** Change the sign of `omega_0` through its slider and find resonance
       again. Then add a preset for $|-x\rangle$ in `spin_presets`. Check its initial Bloch vector before playing.
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
