import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Newns–Anderson model: hydrogen on a transition-metal surface

    An adsorbate level $|a\rangle$ (H $1s$) of energy $\varepsilon_a$ couples to
    the metal.  The coupling is the **self-energy**
    $\Sigma_a(\varepsilon)=\Lambda(\varepsilon)-i\,\Delta(\varepsilon)$, with two
    contributions to $\Delta$:

    - a **d-band** — a semi-elliptic band of half-width $D=W/2$ and coupling
      $\Delta_d$:  $\;\Delta_d\sqrt{1-(\varepsilon/D)^2}$ for $|\varepsilon|\le D$,
      with $\Lambda(\varepsilon)=\Delta_d\,\varepsilon/D$ inside the band;
    - an **s-band** — broad and featureless, giving a **constant** $\Delta_s$.

    $$
    \Delta(\varepsilon)=\underbrace{\Delta_d\sqrt{1-(\varepsilon/D)^2}}_{\text{d-band}}
    \;+\;\underbrace{\Delta_s}_{\text{s-band}},\qquad
    \rho_a(\varepsilon)=\frac{1}{\pi}\,
    \frac{\Delta(\varepsilon)}
    {\bigl(\varepsilon-\varepsilon_a-\Lambda(\varepsilon)\bigr)^2+\Delta(\varepsilon)^2}.
    $$

    A solution of $\varepsilon-\varepsilon_a-\Lambda(\varepsilon)=0$ is a **bound
    state** iff $\Delta(\varepsilon^\*)=0$ (a δ-function of weight
    $Z=1/|1-\Lambda'|$ in $\rho_a$), otherwise a **resonance** (width
    $\sim\Delta(\varepsilon^\*)$).  Hence:

    - $\Delta_s=0$ → genuine **bound states** outside the d-band (δ-functions);
    - $\Delta_s>0$ → the s-band keeps $\mathrm{Im}\,\Sigma\neq0$ everywhere, so
      **only resonances** remain.

    **Drag the coloured points** to explore weak vs. strong coupling and the
    role of the s-band.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import anywidget
    import traitlets

    return anywidget, mo, np, traitlets


@app.cell
def _(np):
    def self_energy(ee, W, delta_d, delta_s=0.0):
        """Adsorbate self-energy for a transition-metal surface.

        The imaginary part (minus Im Sigma = Delta) has two contributions:
          * d-band: a semi-elliptic band of half-width W/2 and coupling delta_d,
          * s-band: a broad, featureless band -> a CONSTANT delta_s.
        The real part (Lambda) comes from the d-band only (the constant s-band
        contributes only to the imaginary part).  Returns (Delta, Lambda)."""
        D = W / 2.0
        x = ee / D
        delta = np.zeros_like(ee)
        lamb = np.zeros_like(ee)
        inside = np.abs(ee) <= D
        below = ee < -D
        above = ee > D
        delta[inside] = delta_d * np.sqrt(1.0 - x[inside] ** 2)
        lamb[inside] = delta_d * x[inside]
        lamb[below] = delta_d * (x[below] + np.sqrt(x[below] ** 2 - 1.0))
        lamb[above] = delta_d * (x[above] - np.sqrt(x[above] ** 2 - 1.0))
        return delta + delta_s, lamb


    def adsorbate_dos(ee, ea, delta, lamb):
        """Projected density of states on the adsorbate level."""
        return (1.0 / np.pi) * delta / ((ee - ea - lamb) ** 2 + delta ** 2)


    def find_features(ea, W, delta_d, delta_s):
        """Solutions of  e - e_a - Lambda(e) = 0, classified by whether Im(Sigma)
        vanishes there:

            bound state  <=>  Delta(e*) = 0   (delta-function pole in rho_a),
            resonance    <=>  Delta(e*) > 0   (finite-width peak).

        With delta_s = 0 the out-of-d-band roots are true bound states; with
        delta_s > 0 the constant s-band keeps Delta > 0 everywhere, so EVERY root
        is a resonance.  Root energies are delta_s-independent (delta_s enters only
        the imaginary part, not Lambda).

        Returns dicts {"energy", "kind", "width", "weight"}:
          * resonance: width  = Delta(e*) / |1 - Lambda'(e*)|,  weight = 0
          * bound    : weight = 1 / |1 - Lambda'(e*)|  (pole residue),  width = 0
        """
        D = W / 2.0
        R = max(1.5, D + delta_d + 0.5, abs(ea) + 0.5) + 2.0
        grid = np.linspace(-R, R, 8000)
        _, lamb = self_energy(grid, W, delta_d, 0.0)  # Lambda is delta_s-independent
        g = grid - ea - lamb
        dlamb = np.gradient(lamb, grid)
        feats = []
        crossings = np.where(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
        for i in crossings:
            a, b, fa = grid[i], grid[i + 1], g[i]
            for _ in range(60):  # bisection refine
                mid = 0.5 * (a + b)
                _, lmid = self_energy(np.array([mid]), W, delta_d, 0.0)
                fm = mid - ea - lmid[0]
                if fa * fm <= 0.0:
                    b = mid
                else:
                    a, fa = mid, fm
            root = 0.5 * (a + b)
            delta_star = float(self_energy(np.array([root]), W, delta_d, delta_s)[0][0])
            slope = abs(1.0 - float(np.interp(root, grid, dlamb)))
            Z = 1.0 / slope if slope > 1e-9 else 1.0
            if delta_star <= 1e-12:  # Im Sigma = 0  ->  genuine bound state
                feats.append({"energy": float(root), "kind": "bound",
                              "width": 0.0, "weight": float(Z)})
            else:
                feats.append({"energy": float(root), "kind": "resonance",
                              "width": float(delta_star * Z), "weight": 0.0})
        return feats


    def plot_frame(n=600):
        """The fixed plot frame: the energy grid and the axis half-ranges.  These do
        NOT depend on the model parameters, so they are built ONCE (not on every
        point/parameter update)."""
        EMAX, XMAX = 3.0, 4.0   # must match EMAX / XMAX in the widget ESM
        ee = np.linspace(-EMAX, EMAX, n)
        return {"ee": ee, "ee_list": ee.round(5).tolist(),
                "emax": float(EMAX), "xmax": float(XMAX)}


    def compute_plotdata(ea, W, delta_d, delta_s, frame):
        """Per-update physics: the parameter-dependent curves and markers, on the
        fixed grid frame["ee"].  rho_a is the TRUE adsorbate DOS (integrates to 1
        together with any bound-state weights) -- it is NOT rescaled."""
        ee = frame["ee"]
        delta, lamb = self_energy(ee, W, delta_d, delta_s)
        rho = adsorbate_dos(ee, ea, delta, lamb)
        return {
            "ee": frame["ee_list"],
            "delta": delta.round(5).tolist(),
            "lamb": lamb.round(5).tolist(),
            "green": (ee - ea).round(5).tolist(),
            "rho": (4.0 * rho).round(5).tolist(),        # x4 DISPLAY factor only (shared qualitative x-axis)
            "emax": frame["emax"],
            "xmax": frame["xmax"],
            "D": float(W / 2.0),
            "bound_mode": bool(delta_s <= 1e-12),        # True: Delta_s=0, bound states
            "features": find_features(ea, W, delta_d, delta_s),
        }

    return compute_plotdata, plot_frame


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **All four parameters are draggable points** — no sliders. Hover any point to
    read its value:

    - 🟢 **$\varepsilon_a$** — adsorbate level; green dot on the axis (drag **↕**)
    - 🔵 **$\Delta_d$** — d-band coupling; blue dot at the tip of the semi-ellipse (drag **↔**)
    - 🟣 **$W$** — d-band width; purple dot at the upper foot of the band (drag **↕**)
    - 🟤 **$\Delta_s$** — s-band coupling (constant); brown dot at the bottom of the
      axis (drag **↔**; set to **0** for genuine bound states)

    The axes freeze while you drag and re-fit when you release.

    ▲ Markers on the energy axis show **bound states** (purple, δ-function arrows
    whose length ∝ weight $Z$, only when $\Delta_s=0$) and **resonances** (red,
    with a shaded width band) — the roots of
    $\varepsilon-\varepsilon_a-\Lambda(\varepsilon)=0$, computed in Python.
    """)
    return


@app.cell
def _(anywidget, compute_plotdata, plot_frame, traitlets):

    _ESM = r"""
    function render({ model, el }) {
      const Wpx = 540, Hpx = 680;
      const m = { top: 24, right: 24, bottom: 22, left: 62 };
      const innerW = Wpx - m.left - m.right;
      const innerH = Hpx - m.top - m.bottom;
      const NS = "http://www.w3.org/2000/svg";
      const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

      // ---- fixed plot scales (no rescaling ever) ----
      const EMAX = 3.0;   // energy (vertical) half-range
      const XMAX = 4.0;   // value (horizontal) half-range
      const yPix = (e) => (EMAX - e) / (2 * EMAX) * innerH;
      const xPix = (v) => (v + XMAX) / (2 * XMAX) * innerW;
      const yInv = (py) => EMAX - (py / innerH) * 2 * EMAX;
      const xInv = (px) => (px / innerW) * 2 * XMAX - XMAX;

      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("width", Wpx);
      svg.setAttribute("height", Hpx);
      svg.style.fontFamily = "system-ui, sans-serif";
      svg.style.touchAction = "none";
      svg.style.userSelect = "none";
      el.appendChild(svg);

      const defs = document.createElementNS(NS, "defs");
      const clip = document.createElementNS(NS, "clipPath");
      clip.setAttribute("id", "naclip");
      const cr = document.createElementNS(NS, "rect");
      cr.setAttribute("x", 0); cr.setAttribute("y", 0);
      cr.setAttribute("width", innerW); cr.setAttribute("height", innerH);
      clip.appendChild(cr); defs.appendChild(clip); svg.appendChild(defs);

      const g = document.createElementNS(NS, "g");
      g.setAttribute("transform", `translate(${m.left},${m.top})`);
      svg.appendChild(g);

      const mk = (tag, attrs) => {
        const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]);
        return e;
      };
      // text with "_x" rendered as a subscript
      const mkText = (x, y, attrs, str) => {
        const t = mk("text", { x, y, ...attrs });
        let buf = "";
        const flush = () => { if (buf) { const s = document.createElementNS(NS, "tspan");
          s.textContent = buf; t.appendChild(s); buf = ""; } };
        for (let i = 0; i < str.length; i++) {
          if (str[i] === "_" && i + 1 < str.length) {
            flush();
            const sub = document.createElementNS(NS, "tspan");
            sub.textContent = str[i + 1];
            sub.setAttribute("baseline-shift", "sub");
            sub.setAttribute("font-size", "0.75em");
            t.appendChild(sub);
            i++;
          } else buf += str[i];
        }
        flush();
        return t;
      };
      const poly = (pts, attrs) => {
        const p = mk("polyline", attrs);
        p.setAttribute("points", pts.map((d) => `${d[0]},${d[1]}`).join(" "));
        return p;
      };
      const areaFill = (vals, ee, attrs) => {
        const pts = [[xPix(0), yPix(ee[0])]];
        for (let i = 0; i < ee.length; i++) pts.push([xPix(vals[i]), yPix(ee[i])]);
        pts.push([xPix(0), yPix(ee[ee.length - 1])]);
        return mk("polygon", { points: pts.map((d) => `${d[0]},${d[1]}`).join(" "),
          "clip-path": "url(#naclip)", ...attrs });
      };
      const P = () => ({ W: model.get("W"), dd: model.get("delta_d"),
        ds: model.get("delta_s"), ea: model.get("epsilon_a") });

      let pd = model.get("plotdata");
      let dragging = false, active = null, lastSave = 0;

      function addHandle(cx, cy, color, id, titleText, labelText, ldx, ldy, anchor) {
        const c = mk("circle", { cx, cy, r: 8, fill: color, stroke: "#222",
          "stroke-width": 1.4,
          cursor: (id === "dd" || id === "ds") ? "ew-resize" : "ns-resize" });
        const t = document.createElementNS(NS, "title");
        t.textContent = titleText; c.appendChild(t);
        c.addEventListener("pointerdown", (ev) => {
          active = id; dragging = true;
          try { svg.setPointerCapture(ev.pointerId); } catch (e) {}
          ev.preventDefault(); ev.stopPropagation();
        });
        g.appendChild(c);
        g.appendChild(mkText(cx + ldx, cy + ldy, { "font-size": 12, fill: color,
          "font-weight": "bold", "text-anchor": anchor || "start" }, labelText));
      }

      function draw() {
        if (!pd) return;
        const p = P();
        const D = pd.D, feats = pd.features || [], ee = pd.ee;
        while (g.firstChild) g.removeChild(g.firstChild);

        // filled areas: Delta (light blue) and rho_a (light red)
        g.appendChild(areaFill(pd.delta, ee, { fill: "#bfdcf3", opacity: 0.8 }));
        g.appendChild(areaFill(pd.rho, ee, { fill: "#f4b6b6", opacity: 0.55 }));

        g.appendChild(mk("line", { x1: xPix(0), y1: 0, x2: xPix(0), y2: innerH,
          stroke: "#999", "stroke-width": 1 }));
        g.appendChild(mk("line", { x1: 0, y1: yPix(0), x2: innerW, y2: yPix(0),
          stroke: "#ddd", "stroke-width": 1 }));

        const nt = 6;
        for (let i = 0; i <= nt; i++) {
          const e = -EMAX + (2 * EMAX) * i / nt, y = yPix(e);
          g.appendChild(mk("line", { x1: -4, y1: y, x2: 0, y2: y, stroke: "#999" }));
          const t = mk("text", { x: -8, y: y + 3, "text-anchor": "end",
            "font-size": 10, fill: "#333" });
          t.textContent = e.toFixed(1); g.appendChild(t);
        }
        const yt = mk("text", { "font-size": 12, "text-anchor": "middle",
          transform: `translate(-46, ${innerH / 2}) rotate(-90)` });
        yt.textContent = "energy ε"; g.appendChild(yt);

        const build = (vals) => { const a = []; for (let i = 0; i < ee.length; i++)
          a.push([xPix(vals[i]), yPix(ee[i])]); return a; };
        const gc = mk("g", { "clip-path": "url(#naclip)" });
        gc.appendChild(poly(build(pd.delta), { fill: "none", stroke: "#1f77b4", "stroke-width": 1.6 }));
        gc.appendChild(poly(build(pd.lamb), { fill: "none", stroke: "#ff7f0e", "stroke-width": 1.6 }));
        gc.appendChild(poly(build(pd.green), { fill: "none", stroke: "#2ca02c",
          "stroke-width": 1.4, "stroke-dasharray": "6,4" }));
        gc.appendChild(poly(build(pd.rho), { fill: "none", stroke: "#d62728", "stroke-width": 1.6 }));
        g.appendChild(gc);

        // decluttered markers: short tick + label (resonance) / delta-arrow (bound)
        feats.forEach((ft) => {
          const y = yPix(ft.energy), bound = ft.kind === "bound";
          const col = bound ? "#6a3d9a" : "#d62728";
          if (bound) {
            const x0 = xPix(0), xt = xPix(ft.weight * XMAX * 0.85);
            g.appendChild(mk("line", { x1: x0, y1: y, x2: xt, y2: y,
              stroke: col, "stroke-width": 2.5 }));
            g.appendChild(mk("path", { d: `M${xt},${y} l-9,-5 l0,10 z`, fill: col }));
            g.appendChild(mkText(xt + 5, y - 5, { "font-size": 10, fill: col,
              "font-weight": "bold" }, `ε=${ft.energy.toFixed(2)}  Z=${ft.weight.toFixed(2)}`));
          } else {
            g.appendChild(mk("line", { x1: 0, y1: y, x2: 10, y2: y,
              stroke: col, "stroke-width": 2 }));
            g.appendChild(mkText(13, y - 4, { "font-size": 10, fill: col },
              `res ε*=${ft.energy.toFixed(2)}`));
          }
        });

        const legend = [["Δ(ε)", "#1f77b4"], ["Λ(ε)", "#ff7f0e"],
          ["ε−ε_a", "#2ca02c"], ["ρ_a", "#d62728"]];
        legend.forEach((it, i) => {
          const ly = 4 + i * 17;
          g.appendChild(mk("line", { x1: innerW - 116, y1: ly, x2: innerW - 96,
            y2: ly, stroke: it[1], "stroke-width": 3 }));
          g.appendChild(mkText(innerW - 92, ly + 4, { "font-size": 11, fill: "#222" }, it[0]));
        });

        // handles.  ea and W labels go to the LEFT of the y-axis (off the curves).
        addHandle(xPix(0), yPix(p.ea), "#2ca02c", "ea",
          `ε_a = ${p.ea.toFixed(3)}  (adsorbate level, drag ↕)`,
          `ε_a=${p.ea.toFixed(2)}`, -13, 4, "end");
        addHandle(xPix(0), yPix(p.W / 2), "#9467bd", "W",
          `W = ${p.W.toFixed(3)}  (d-band width, drag ↕)`,
          `W=${p.W.toFixed(2)}`, -13, -6, "end");
        addHandle(xPix(p.dd + p.ds), yPix(0), "#1f77b4", "dd",
          `Δ_d = ${p.dd.toFixed(3)}  (d-band coupling, drag ↔)`,
          `Δ_d=${p.dd.toFixed(2)}`, -12, -10, "end");
        addHandle(xPix(p.ds), yPix(-0.9 * EMAX), "#8c564b", "ds",
          `Δ_s = ${p.ds.toFixed(3)}  (s-band coupling; = 0 → bound states, drag ↔)`,
          `Δ_s=${p.ds.toFixed(3)}`, 13, 4, "start");
      }

      svg.addEventListener("pointermove", (ev) => {
        if (!dragging) return;
        const rect = svg.getBoundingClientRect();
        const px = clamp(ev.clientX - rect.left - m.left, 0, innerW);
        const py = clamp(ev.clientY - rect.top - m.top, 0, innerH);
        const yv = yInv(py), xv = xInv(px);
        const ds = model.get("delta_s");
        if (active === "ea") model.set("epsilon_a", clamp(yv, -EMAX, EMAX));
        else if (active === "dd") model.set("delta_d", clamp(xv - ds, 0.05, Math.min(3, XMAX - ds - 0.1)));
        else if (active === "W") model.set("W", 2 * clamp(yv, 0.05, 2.85));
        else if (active === "ds") model.set("delta_s", clamp(xv, 0, 2.0));
        draw();
        const now = Date.now();
        if (now - lastSave > 33) { lastSave = now; model.save_changes(); }
      });
      const endDrag = (ev) => {
        if (!dragging) return;
        dragging = false; active = null;
        try { svg.releasePointerCapture(ev.pointerId); } catch (e) {}
        model.save_changes();
      };
      svg.addEventListener("pointerup", endDrag);
      svg.addEventListener("pointercancel", endDrag);

      model.on("change:plotdata", () => { pd = model.get("plotdata"); draw(); });
      ["change:W", "change:delta_d", "change:delta_s", "change:epsilon_a"]
        .forEach((ev) => model.on(ev, draw));
      draw();
    }
    export default { render };
    """


    class NAWidget(anywidget.AnyWidget):
        W = traitlets.Float(1.0).tag(sync=True)          # d-band width
        delta_d = traitlets.Float(2.0).tag(sync=True)    # d-band coupling
        delta_s = traitlets.Float(0.0).tag(sync=True)    # s-band coupling (constant)
        epsilon_a = traitlets.Float(-0.4).tag(sync=True)
        plotdata = traitlets.Dict().tag(sync=True)
        _esm = _ESM


    na_widget = NAWidget()
    _frame = plot_frame()  # constant grid/scales, built once (not per update)


    def _recompute(change=None):
        try:
            na_widget.plotdata = compute_plotdata(
                na_widget.epsilon_a, na_widget.W, na_widget.delta_d, na_widget.delta_s, _frame
            )
        except Exception:
            pass


    _recompute()
    na_widget.observe(_recompute, names=["W", "delta_d", "delta_s", "epsilon_a"])
    na_widget

    return


if __name__ == "__main__":
    app.run()
