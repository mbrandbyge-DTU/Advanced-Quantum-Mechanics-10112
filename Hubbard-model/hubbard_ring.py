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
    # Hubbard model: two electrons on a ring
    **DTU 10112 · Problem set "Hubbard model on a ring"** · one ↑ and one ↓ electron on $L$ sites, energies in units of $t$:
    $H=-t\sum_{n,s}\big(c^\dagger_{n+1,s}c_{n,s}+c^\dagger_{n,s}c_{n+1,s}\big)+U\sum_n c^\dagger_{n\uparrow}c_{n\uparrow}c^\dagger_{n\downarrow}c_{n\downarrow}$.
    In the zero-momentum basis $|k\uparrow,-k\downarrow\rangle$: $\;\langle k|H|k'\rangle=2\varepsilon_k\,\delta_{kk'}+U/L$, with $\varepsilon_k=-2t\cos k$.
    Make the predictions in the section **Predict → try → explain** at the bottom, then use the two experiments below.
    """)
    return


@app.cell(hide_code=True)
def _(spectrum_L, spectrum_U, spectrum_panel, spectrum_range, spectrum_state):
    _widgets = {"L": spectrum_L, "U": spectrum_U, "range": spectrum_range, "n": spectrum_state}
    spectrum_panel(_widgets, {_k: _w.value for _k, _w in _widgets.items()})
    return


@app.cell(hide_code=True)
def _(hole_controls, hole_panel):
    hole_panel(hole_controls, hole_controls.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Physics — small NumPy routines

    **Basis.** $k_p=2\pi p/L$, $p=0,\dots,L-1$, and $|k_p\rangle\equiv|k_p\uparrow,-k_p\downarrow\rangle=\psi^\dagger_{k_p\uparrow}\psi^\dagger_{-k_p\downarrow}|0\rangle$.
    $H_{TB}$ is diagonal with $\varepsilon_k+\varepsilon_{-k}=-4t\cos k$; the on-site term scatters the pair
    $(k,-k)\to(k',-k')$ with the same amplitude $U/L$ for every $k,k'$. A state is
    $|\Psi\rangle=\sum_k\phi(k)\,|k\rangle$.

    **Real space.** $\Psi(n_1\!\uparrow,n_2\!\downarrow)=\frac1L\sum_k e^{ik(n_1-n_2)}\phi(k)$ depends only on
    $r=n_1-n_2$ (mod $L$). $P(r)=L\,|\Psi(r)|^2$ is the probability that the ↓ electron sits $r$ sites
    from the ↑ electron; $\sum_r P(r)=1$, and $P(0)=\langle\sum_n n_{n\uparrow}n_{n\downarrow}\rangle$ is the
    **double occupancy**.

    **Spin.** Swapping $k\to-k$ exchanges the momenta of the two electrons. $H$ commutes with this
    exchange $X$, so every eigenstate has $\phi(-k)=\pm\phi(k)$: $+$ is a **singlet** ($S=0$, symmetric
    in space), $-$ is the $S_z=0$ member of a **triplet** ($S=1$, antisymmetric in space).
    """)
    return


@app.cell
def _(np):
    # PHYSICS ONLY: no widgets or plotting. Change the Hamiltonian here; every
    # plot in the notebook follows automatically.
    def k_values(L):
        """Allowed crystal momenta k_p = 2*pi*p/L on a ring of L sites."""
        return 2 * np.pi * np.arange(L) / L

    def pair_hamiltonian(L, U, t=1.0):
        """H in the basis |k_p up, -k_p down>, p = 0..L-1 (total momentum zero)."""
        k = k_values(L)
        H = np.zeros((L, L))
        for p in range(L):
            H[p, p] = -2 * t * np.cos(k[p]) - 2 * t * np.cos(-k[p])  # eps(k) + eps(-k)
        for p in range(L):
            for q in range(L):
                H[p, q] += U / L                                      # on-site repulsion
        # STUDENT EDIT: add more terms here, e.g. a nearest-neighbour repulsion V.
        return H

    def exchange_matrix(L):
        """X|k up, -k down> = |-k up, k down>: swaps the electrons' momenta."""
        X = np.zeros((L, L))
        for p in range(L):
            X[(-p) % L, p] = 1.0
        return X

    return exchange_matrix, k_values, pair_hamiltonian


@app.cell
def _(exchange_matrix, np, pair_hamiltonian):
    def solve_pair(L, U, t=1.0):
        """Eigenenergies E[n], eigenvectors phi[:, n] (sorted by energy) and spin S[n].

        At U = 0, the levels +k and -k are degenerate and eigh may return a mixture
        of singlet and triplet. Inside each degenerate group we therefore also
        diagonalize the exchange X, which commutes with H.
        """
        H = pair_hamiltonian(L, U, t)
        X = exchange_matrix(L)
        E, phi = np.linalg.eigh(H)
        start = 0
        while start < L:
            stop = start + 1
            while stop < L and abs(E[stop] - E[start]) < 1e-9:
                stop += 1
            if stop - start > 1:
                block = phi[:, start:stop]
                _, R = np.linalg.eigh(block.T @ X @ block)
                phi[:, start:stop] = block @ R
            start = stop
        S = np.zeros(L, dtype=int)
        for n in range(L):
            parity = phi[:, n] @ X @ phi[:, n]       # +1 or -1
            S[n] = 0 if parity > 0 else 1
            # Fix the arbitrary overall sign: largest component positive.
            biggest = np.argmax(np.abs(phi[:, n]))
            phi[:, n] = phi[:, n] * np.sign(phi[biggest, n])
        return E, phi, S

    def real_space(phi_n, L):
        """Psi(r) = (1/L) sum_k exp(i k r) phi(k) for separations r = n1 - n2 = 0..L-1."""
        k = 2 * np.pi * np.arange(L) / L
        Psi = np.zeros(L, dtype=complex)
        for r in range(L):
            for p in range(L):
                Psi[r] += np.exp(1j * k[p] * r) * phi_n[p] / L
        return Psi

    def separation_probability(phi_n, L):
        """P(r) = L |Psi(r)|^2; P(0) is the double occupancy."""
        return L * np.abs(real_space(phi_n, L)) ** 2

    return separation_probability, solve_pair


@app.cell
def _(np, separation_probability, solve_pair):
    def spectrum_vs_U(L, U_values):
        """Energies of all eigenstates on a grid of U, split into singlets and triplets."""
        singlets, triplets = [], []
        for U in U_values:
            E, _, S = solve_pair(L, U)
            singlets.append(E[S == 0])
            triplets.append(E[S == 1])
        return np.array(singlets), np.array(triplets)

    def ground_state_vs_U(L, U_values):
        """Ground-state energy E0(U) and double occupancy P(r=0) on a grid of U."""
        E0, double_occ = [], []
        for U in U_values:
            E, phi, _ = solve_pair(L, U)
            E0.append(E[0])
            double_occ.append(separation_probability(phi[:, 0], L)[0])
        return np.array(E0), np.array(double_occ)

    return ground_state_vs_U, spectrum_vs_U


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interface and drawing — optional code

    The following code cells are collapsed. The complete experiments appear above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # INTERFACE ONLY: controls for the spectrum experiment that do not depend on each other.
    U_ranges = {"0 to 4 (problem set)": (0.0, 4.0), "−8 to 8": (-8.0, 8.0), "−10 to 50": (-10.0, 50.0)}
    spectrum_L = mo.ui.slider(2, 16, step=1, value=6, label="Ring size L", show_value=True)
    spectrum_U = mo.ui.slider(-10, 50, step=0.1, value=2.0, label="U / t", show_value=True)
    spectrum_range = mo.ui.dropdown(list(U_ranges), value="0 to 4 (problem set)", label="U axis")
    return U_ranges, spectrum_L, spectrum_U, spectrum_range


@app.cell(hide_code=True)
def _(mo, spectrum_L):
    # INTERFACE ONLY: own cell, so the slider is rebuilt whenever L changes.
    # The panel shows these widgets themselves, NOT clones inside a mo.ui.dictionary:
    # moving a dictionary's clone of spectrum_L would never re-run this cell.
    spectrum_state = mo.ui.slider(0, spectrum_L.value - 1, step=1, value=0,
                                  label="Eigenstate n (0 = ground)", show_value=True)
    return (spectrum_state,)


@app.cell(hide_code=True)
def _(U_ranges, mo):
    # INTERFACE ONLY: independent controls for the correlation-hole experiment.
    hole_controls = mo.ui.dictionary({
        "L": mo.ui.slider(2, 16, step=1, value=6, label="Ring size L", show_value=True),
        "U": mo.ui.slider(-10, 50, step=0.1, value=2.0, label="U / t", show_value=True),
        "range": mo.ui.dropdown(list(U_ranges), value="−8 to 8", label="U axis"),
        "log": mo.ui.checkbox(value=False, label="log scale for P(0)"),
    })
    return (hole_controls,)


@app.cell(hide_code=True)
def _(go, make_subplots, np):
    # DRAWING ONLY: the three plots of the spectrum experiment.
    SINGLET, TRIPLET, REF = "#2b6cb0", "#dd6b20", "#a0aec0"

    def spectrum_figure(U_grid, singlets, triplets, U, E_sel, k_over_pi, phi_sel,
                        P_sel, P_ref, spin_sel):
        fig = make_subplots(rows=1, cols=3, column_widths=[0.42, 0.29, 0.29],
                            horizontal_spacing=0.07, subplot_titles=(
                                "Energies, total momentum 0",
                                "Pair amplitude φ(k)",
                                "Separation probability P(r)"))
        for j in range(singlets.shape[1]):
            fig.add_trace(go.Scatter(x=U_grid, y=singlets[:, j], mode="lines",
                line=dict(color=SINGLET, width=2), name="singlet S=0",
                legendgroup="s", showlegend=(j == 0)), row=1, col=1)
        for j in range(triplets.shape[1]):
            fig.add_trace(go.Scatter(x=U_grid, y=triplets[:, j], mode="lines",
                line=dict(color=TRIPLET, width=2, dash="dash"), name="triplet S=1",
                legendgroup="t", showlegend=(j == 0)), row=1, col=1)
        color = SINGLET if spin_sel == 0 else TRIPLET
        fig.add_vline(x=U, line=dict(color="#4a5568", width=1, dash="dot"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[U], y=[E_sel], mode="markers", name="state n",
            marker=dict(color=color, size=12, symbol="star", line=dict(color="black", width=1))),
            row=1, col=1)

        fig.add_trace(go.Bar(x=k_over_pi, y=phi_sel, marker_color=color, showlegend=False,
                             width=1.6 / len(k_over_pi)), row=1, col=2)

        r = np.arange(len(P_sel))
        fig.add_trace(go.Bar(x=r, y=P_sel, marker_color=color, showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=r, y=P_ref, mode="markers", name="same level, U=0",
            marker=dict(color=REF, size=9, symbol="line-ew-open", line=dict(width=3))),
            row=1, col=3)

        fig.update_xaxes(title="U / t", range=[U_grid[0], U_grid[-1]], row=1, col=1)
        fig.update_yaxes(title="E / t", row=1, col=1)
        fig.update_xaxes(title="k / π", range=[-1.1, 1.1], row=1, col=2)
        fig.update_yaxes(range=[-1.05, 1.05], row=1, col=2)
        fig.update_xaxes(title="r = n₁ − n₂ (mod L)", dtick=1, row=1, col=3)
        fig.update_yaxes(range=[0, max(0.3, 1.1 * max(P_sel.max(), P_ref.max()))], row=1, col=3)
        fig.update_layout(template="plotly_white", height=340, font=dict(size=11),
                          margin=dict(l=45, r=10, t=60, b=40), bargap=0.15,
                          legend=dict(orientation="h", y=1.22, x=0, font=dict(size=10)))
        return fig

    def hole_figure(L, P, P_free, U_grid, E0_grid, D_grid, U, D, log):
        fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                            horizontal_spacing=0.08, subplot_titles=(
                                "Where is ↓ when ↑ sits at site 1?",
                                "Double occupancy P(0) of the ground state"))
        # Ring: site 1 at the top, numbered clockwise. Area of a blue disc is P(r).
        angle = np.pi / 2 - 2 * np.pi * np.arange(L) / L
        x, y = np.cos(angle), np.sin(angle)
        circle = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(x=np.cos(circle), y=np.sin(circle), mode="lines",
            line=dict(color="#cbd5e0", width=2), hoverinfo="skip", showlegend=False), row=1, col=1)
        size = lambda p: 6 + 90 * np.sqrt(p)
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="U = 0",
            marker=dict(size=size(P_free), color="rgba(0,0,0,0)",
                        line=dict(color=REF, width=1.5)), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="P(r) at this U",
            marker=dict(size=size(P), color=SINGLET, opacity=0.75),
            text=[f"site {n+1}, r = {n}: P = {P[n]:.3f}" for n in range(L)],
            hoverinfo="text"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0], y=[1.32], mode="text", text=["↑ electron fixed here"],
            textfont=dict(color="#c53030", size=11), hoverinfo="skip", showlegend=False),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=[0], y=[1.0], mode="markers", hoverinfo="skip",
            marker=dict(symbol="triangle-up", size=14, color="#c53030"), showlegend=False),
            row=1, col=1)
        fig.update_xaxes(visible=False, range=[-1.55, 1.55], row=1, col=1)
        fig.update_yaxes(visible=False, range=[-1.45, 1.5], scaleanchor="x", row=1, col=1)

        # Hellmann-Feynman: dE0/dU = <sum_n n_up n_down> = P(0). Sparse dots so the line shows through.
        slope = np.gradient(E0_grid, U_grid)
        every = max(1, len(U_grid) // 40)
        fig.add_trace(go.Scatter(x=U_grid, y=D_grid, mode="lines", name="P(0) from Ψ(r)",
            line=dict(color=SINGLET, width=2.5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=U_grid[::every], y=slope[::every], mode="markers",
            name="dE₀/dU", marker=dict(color="#2f855a", size=5)), row=1, col=2)
        fig.add_hline(y=1 / L, line=dict(color=REF, dash="dot"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[U], y=[D], mode="markers", showlegend=False,
            marker=dict(color=SINGLET, size=12, symbol="star", line=dict(color="black", width=1))),
            row=1, col=2)
        fig.update_xaxes(title="U / t", range=[U_grid[0], U_grid[-1]], row=1, col=2)
        if log:
            fig.update_yaxes(title="P(0)", type="log", range=[-5, 0.02],
                             exponentformat="power", row=1, col=2)
        else:
            fig.update_yaxes(title="P(0)", range=[0, 1.02], row=1, col=2)
        fig.update_layout(template="plotly_white", height=360, font=dict(size=11),
                          margin=dict(l=20, r=10, t=60, b=40),
                          legend=dict(orientation="h", y=1.22, x=0, font=dict(size=10)))
        return fig

    return hole_figure, spectrum_figure


@app.cell(hide_code=True)
def _(
    U_ranges,
    k_values,
    mo,
    np,
    separation_probability,
    solve_pair,
    spectrum_figure,
    spectrum_vs_U,
):
    # INTERFACE ADAPTER: read controls, call the physics, assemble ONE output.
    def spectrum_panel(controls, values):
        L, U = values["L"], values["U"]
        # The n slider is rebuilt when L changes but can report a stale value for one tick.
        n = int(np.clip(values["n"], 0, L - 1))
        lo, hi = U_ranges[values["range"]]
        U_grid = np.linspace(min(lo, U), max(hi, U), 241)
        singlets, triplets = spectrum_vs_U(L, U_grid)

        E, phi, S = solve_pair(L, U)
        P_sel = separation_probability(phi[:, n], L)
        # U = 0 reference: the same level, i.e. same spin and same rank among that spin.
        rank = int(np.sum(S[:n] == S[n]))
        E0, phi0, S0 = solve_pair(L, 0.0)
        n0 = np.flatnonzero(S0 == S[n])[rank]
        P_ref = separation_probability(phi0[:, n0], L)

        k = k_values(L)
        k_over_pi = np.angle(np.exp(1j * k)) / np.pi        # fold into (-1, 1]
        k_over_pi[np.isclose(k_over_pi, -1)] = 1.0
        order = np.argsort(k_over_pi)

        fig = spectrum_figure(U_grid, singlets, triplets, U, E[n], k_over_pi[order],
                              phi[order, n], P_sel, P_ref, S[n])
        chart = mo.ui.plotly(fig, config={"displayModeBar": False, "responsive": True})

        spin = "singlet (S = 0)" if S[n] == 0 else "triplet (S = 1)"
        details = (f"State n = {n}: E = {E[n]:.4f} t, {spin}; double occupancy P(0) = {P_sel[0]:.4f}. "
                   f"Number of singlets: {np.sum(S == 0)}, triplets: {np.sum(S == 1)}. "
                   f"Gray marks in the right panel: the same level at U = 0.")
        panel = mo.vstack([
            mo.md("### Experiment 1 — spectrum and eigenstates"),
            mo.hstack([controls["L"], controls["U"], controls["range"], controls["n"]],
                      wrap=True, gap=0.8, justify="start"),
            chart,
            mo.md(details),
        ], gap=0.3).style({"border": "1px solid #dce3eb", "border-radius": "12px",
                           "padding": "12px", "margin-bottom": "14px"})
        return panel

    return (spectrum_panel,)


@app.cell(hide_code=True)
def _(
    U_ranges,
    ground_state_vs_U,
    hole_figure,
    mo,
    np,
    separation_probability,
    solve_pair,
):
    # INTERFACE ADAPTER: read controls, call the physics, assemble ONE output.
    def hole_panel(controls, values):
        L, U = values["L"], values["U"]
        E, phi, _ = solve_pair(L, U)
        P = separation_probability(phi[:, 0], L)
        _, phi_free, _ = solve_pair(L, 0.0)
        P_free = separation_probability(phi_free[:, 0], L)
        lo, hi = U_ranges[values["range"]]
        U_grid = np.linspace(min(lo, U), max(hi, U), 401)
        E0_grid, D_grid = ground_state_vs_U(L, U_grid)

        fig = hole_figure(L, P, P_free, U_grid, E0_grid, D_grid, U, P[0], values["log"])
        chart = mo.ui.plotly(fig, config={"displayModeBar": False, "responsive": True})

        details = (f"Ground state at U/t = {U:.1f}: E₀ = {E[0]:.4f} t, "
                   f"P(0) = {P[0]:.4f} (uncorrelated value 1/L = {1/L:.4f}), "
                   f"most likely separation r = {int(np.argmax(P))}. "
                   "Open circles on the ring: U = 0. Green dots: slope dE₀/dU of the ground-state energy.")
        panel = mo.vstack([
            mo.md("### Experiment 2 — the correlation hole"),
            mo.hstack([controls["L"], controls["U"], controls["range"], controls["log"]],
                      wrap=True, gap=0.8, justify="start"),
            chart,
            mo.md(details),
        ], gap=0.3).style({"border": "1px solid #dce3eb", "border-radius": "12px",
                           "padding": "12px", "margin-bottom": "14px"})
        return panel

    return (hole_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict → try → explain

    1. **No interaction.** Before touching anything: for $L=6$ and $U=0$, list the pair energies
       $-4t\cos k$. Which are degenerate, and why? Check with Experiment 1 at $U=0$.
    2. **Switch on $U$.** Predict which levels move when $U$ grows from 0 to 4. Then look: how many
       states change, how many stay put? Step through $n$ and use the $\phi(k)$ panel to decide which are
       singlets and which triplets. Why are the triplet energies so boring?
    3. **The matrix element $U/L$.** Show $\langle k\uparrow,-k\downarrow|H_C|k'\uparrow,-k'\downarrow\rangle=U/L$ on paper.
       Then find it in `pair_hamiltonian`. What happens to the triplets if the interaction is not
       on-site (task 7)?
    4. **The correlation hole.** In Experiment 2, predict the ground-state $P(r)$ at $U=0$. Increase $U$:
       how does the depth of the hole at $r=0$ depend on $U$? Why does it not go straight to zero?
       Compare $U/t=2$, 10, 50, then switch on the log scale (U axis −10 to 50): how fast does $P(0)$
       fall at large $U$?
    5. **Energy versus wavefunction.** The ground-state energy changes only modestly with $U$, while
       $P(0)$ changes a lot. Compare the blue line with the green dots and explain with the
       Hellmann–Feynman theorem. What do the green dots tell you about the curvature of $E_0(U)$?
    6. **Attraction.** Set $U<0$. What happens to the pair? Predict $P(0)$ as $U\to-\infty$.
       Does the answer depend on $L$?
    7. **Edit the physics.** Add a nearest-neighbour repulsion $V$ between the two electrons. In this basis
       it gives $\langle k|H_V|k'\rangle=\frac{2V}{L}\cos(k-k')$; add in `pair_hamiltonian`
       (before `return H`):
       ```python
       V = 1.0
       for p in range(L):
           for q in range(L):
               H[p, q] += 2 * V / L * np.cos(k[p] - k[q])
       ```
       Predict first: do the triplet levels still stay flat as $U$ changes? Do they move compared with $V=0$?
       Where does the ground-state $P(r)$ now peak?

    **AI as a checking partner:** derive the $U/L$ matrix element yourself, then ask an AI assistant
    to derive the nearest-neighbour matrix element of task 7. Check its answer against the
    notebook: do the triplets move as you predicted?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Discussion notes — open after making predictions": mo.md(r"""
    **Spectrum.** At $U=0$ the energies are $-4t\cos k$: for $L=6$, $-4$, $-2$ (twice, $k=\pm\pi/3$),
    $+2$ (twice, $k=\pm2\pi/3$) and $+4$. Each degenerate pair splits into a symmetric combination
    $\phi(k)=\phi(-k)$ (singlet) and an antisymmetric one $\phi(k)=-\phi(-k)$ (triplet). For $L=6$ there are
    4 singlets and 2 triplets. The ground state and the top state move smoothly; of the two degenerate
    pairs, the singlet moves up and the triplet stays.

    **Boring triplets.** In this basis $H_C=U\,|u\rangle\langle u|$ with $|u\rangle=(1,\dots,1)/\sqrt L$, so only
    the component $\sum_k\phi(k)=L\,\Psi(r=0)$ feels $U$. For a triplet this sum vanishes: the spatial
    wavefunction is antisymmetric, so the two electrons are never on the same site (Pauli), and
    the on-site repulsion has nothing to act on. Triplet energies and wavefunctions are independent of $U$.
    A singlet energy solves $\frac LU=\sum_k\frac{1}{E+4t\cos k}$, so it is pinned between the $U=0$ levels.

    **Correlation hole.** $P(0)$ starts at $1/L$ for $U=0$ (uncorrelated plane wave $k=0$) and decreases
    smoothly with $U$. The kinetic energy prefers the long-wavelength, uniform state; the repulsion
    prefers no double occupancy. Removing the double occupancy completely would need a wavefunction
    built from high-$k$ components, which costs kinetic energy of order $t$. Second-order perturbation
    theory at large $U$ gives $P(0)\propto(t/U)^2$, so the hole deepens gradually and never reaches zero
    at finite $U$.

    **Hellmann–Feynman.** $dE_n/dU=\langle n|\sum_m n_{m\uparrow}n_{m\downarrow}|n\rangle=P_n(0)$.
    The slope of $E_0(U)$ *is* the double occupancy, so $E_0$ saturates as the hole forms. Since $P(0)$ decreases,
    $E_0(U)$ is concave. Triplets: slope zero, $P(0)=0$.

    **Attraction.** For $U<0$ the ground state becomes a bound pair localized in $r$: $P(0)\to1$ as $U\to-\infty$,
    and $E_0\approx U$.

    **Nearest-neighbour $V$.** Now $\Psi(\pm1)\neq0$ matters, and triplets do have weight there, so their
    energies shift with $V$ (still independent of $U$). For large $U$ and $V$ the ground state pushes the
    ↓ electron to $|r|\ge2$.
    """)})
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
