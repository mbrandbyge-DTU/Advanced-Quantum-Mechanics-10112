# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=1.26", "anywidget>=0.9", "traitlets>=5.0"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import anywidget
    import traitlets

    return anywidget, mo, np, traitlets


@app.cell(hide_code=True)
def _(mo):
    # TODO before final release: do one more pass over the hidden cells below
    # and strip any remaining internal/programming notes (library or CDN
    # mentions, implementation rationale, etc.) so only physics- and
    # teaching-facing comments remain anywhere in this notebook.
    mo.md(r"""
    # Spin-1/2 dynamics
    **DTU 10112 · Time-dependent phenomena · Ballentine, pp. 332–349** · $\hbar=1$,
    $\omega_i=\gamma B_i$: $H(t)=-\frac12[\omega_0\sigma_z+\omega_1\cos(\omega t)\sigma_x+\omega_1\sin(\omega t)\sigma_y]$.
    Try the two experiments below; the physics and full write-up follow underneath
    them. Drag the sphere any time, including while playing, to look from any angle.
    """)
    return


@app.cell(hide_code=True)
def _(make_controls):
    precession_controls = make_controls(rotating=False)
    return (precession_controls,)


@app.cell(hide_code=True, expand_output=True)
def _(precession_controls, simulation_panel):
    simulation_panel(precession_controls, precession_controls.value, rotating=False)
    return


@app.cell(hide_code=True)
def _(make_controls):
    resonance_controls = make_controls(rotating=True)
    return (resonance_controls,)


@app.cell(hide_code=True, expand_output=True)
def _(resonance_controls, simulation_panel):
    simulation_panel(resonance_controls, resonance_controls.value, rotating=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Physics — small NumPy routines""")
    return


@app.cell
def _(np):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

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

    The following code cells are collapsed. The complete experiments appear above.
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
        return mo.ui.dictionary(controls)

    return (make_controls,)


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    # DRAWING ONLY: a from-scratch three.js scene. Geometry is created once;
    # each frame only updates a handful of numbers (arrow orientation/length,
    # trail draw range, 2D canvas redraw) — lightweight at runtime.
    # three.js is dynamically imported inside render() (not as a static ES
    # import) so the "Loading 3D view…" placeholder actually appears on screen
    # immediately instead of the widget staying blank while the CDN fetch
    # resolves.
    ESM = r"""
    function toThree(THREE, v) { return new THREE.Vector3(v[0], v[2], v[1]); }

    function makeTextSprite(THREE, text) {
      const canvas = document.createElement("canvas");
      canvas.width = 128; canvas.height = 128;
      const ctx = canvas.getContext("2d");
      ctx.font = "bold 92px Georgia, serif";
      ctx.fillStyle = "#243247";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 64, 68);
      const tex = new THREE.CanvasTexture(canvas);
      tex.minFilter = THREE.LinearFilter;
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
      const sprite = new THREE.Sprite(mat);
      sprite.scale.set(0.22, 0.22, 1);
      return sprite;
    }

    function makeArrow(THREE, color) {
      const group = new THREE.Group();
      const shaftGeom = new THREE.CylinderGeometry(0.022, 0.022, 1, 14);
      shaftGeom.translate(0, 0.5, 0);
      const headGeom = new THREE.ConeGeometry(0.065, 0.18, 18);
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.05 });
      const shaft = new THREE.Mesh(shaftGeom, mat);
      const head = new THREE.Mesh(headGeom, mat);
      group.add(shaft); group.add(head);
      group.userData = { shaft, head };
      return group;
    }

    function updateArrow(THREE, group, vec3) {
      const len = vec3.length();
      if (len < 1e-6) { group.visible = false; return; }
      group.visible = true;
      const dir = vec3.clone().normalize();
      const headLen = Math.min(0.2, 0.35 * len);
      const shaftLen = Math.max(len - headLen, 0.001);
      group.userData.shaft.scale.set(1, shaftLen, 1);
      group.userData.head.position.set(0, shaftLen + headLen / 2, 0);
      group.userData.head.scale.set(1, headLen / 0.18, 1);
      group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    }

    function makeCircle(THREE, plane) {
      const pts = [];
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        if (plane === "xy") pts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0));
        if (plane === "xz") pts.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
        if (plane === "yz") pts.push(new THREE.Vector3(0, Math.cos(a), Math.sin(a)));
      }
      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color: 0xa5b3bf });
      return new THREE.Line(geom, mat);
    }

    async function render({ model, el }) {
      el.innerHTML = `<div style="padding:50px 20px; text-align:center; color:#8a94a6; font-family: Georgia, serif;">Loading 3D view…</div>`;

      const [THREE, { OrbitControls }] = await Promise.all([
        import("https://esm.sh/three@0.160.0"),
        import("https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js"),
      ]);

      const legendRow = (color, label) =>
        `<span style="margin-right:12px;"><span style="display:inline-block;width:9px;height:9px;` +
        `background:${color};border-radius:2px;margin-right:4px;"></span>${label}</span>`;
      el.innerHTML = `
        <div style="display:flex; gap:14px; align-items:flex-start; font-family: Georgia, serif; flex-wrap: wrap;">
          <div>
            <div style="font-size:12px; color:#243247; margin-bottom:3px; display:flex; justify-content:space-between;">
              <span class="scene-title"></span>
              <span>${legendRow("#dd8b22", "B-field axis")}${legendRow("#235ba8", "Spin ⟨σ⟩")}</span>
            </div>
            <div class="scene-container" style="width:300px; height:300px; cursor: grab;"></div>
            <div style="margin-top:8px; display:flex; align-items:center; gap:6px;">
              <button class="play-btn">&#9654; Play</button>
              <button class="pause-btn">Pause</button>
              <button class="reset-btn">Reset</button>
              <input type="range" class="time-slider" min="0" max="200" value="0" style="flex:1; min-width:80px;">
              <span class="time-label" style="font-size:12px; color:#243247;">t = 0.00</span>
            </div>
          </div>
          <div>
            <div style="font-size:12px; color:#243247; margin-bottom:3px;">
              Measurement probabilities &nbsp;
              ${legendRow("#235ba8", "P(−z)")}${legendRow("#b04d86", "P(+x), lab")}
            </div>
            <canvas class="prob-canvas" width="300" height="300" style="border:1px solid #dce3eb; border-radius:8px;"></canvas>
          </div>
        </div>
      `;
      el.querySelector(".scene-title").textContent = model.get("title") || "";

      const sceneContainer = el.querySelector(".scene-container");
      const probCanvas = el.querySelector(".prob-canvas");
      const playBtn = el.querySelector(".play-btn");
      const pauseBtn = el.querySelector(".pause-btn");
      const resetBtn = el.querySelector(".reset-btn");
      const slider = el.querySelector(".time-slider");
      const timeLabel = el.querySelector(".time-label");

      const times = model.get("times");
      const rArr = model.get("r");
      const bArr = model.get("b_axis");
      const pDown = model.get("p_down");
      const pX = model.get("p_x");
      const N = times.length;
      slider.max = String(N - 1);

      const width = 300, height = 300;
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      sceneContainer.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 100);
      camera.position.set(2.7, 2.0, 2.7);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(0, 0, 0);

      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
      dirLight.position.set(2.5, 3.5, 2);
      scene.add(dirLight);

      const sphereGeom = new THREE.SphereGeometry(1, 48, 32);
      const sphereMat = new THREE.MeshBasicMaterial({ color: 0xa8c4dc, transparent: true, opacity: 0.12 });
      scene.add(new THREE.Mesh(sphereGeom, sphereMat));
      ["xy", "xz", "yz"].forEach((p) => scene.add(makeCircle(THREE, p)));

      const axisMat = new THREE.LineBasicMaterial({ color: 0x555555 });
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((dir) => {
        const a = toThree(THREE, dir.map((c) => -1.15 * c));
        const b = toThree(THREE, dir.map((c) => 1.18 * c));
        const geom = new THREE.BufferGeometry().setFromPoints([a, b]);
        scene.add(new THREE.Line(geom, axisMat));
      });
      const labelPos = { x: [1.3, 0, 0], y: [0, 1.3, 0], z: [0, 0, 1.3] };
      Object.entries(labelPos).forEach(([name, pos]) => {
        const s = makeTextSprite(THREE, name);
        s.position.copy(toThree(THREE, pos));
        scene.add(s);
      });

      const spinArrow = makeArrow(THREE, 0x235ba8);
      const bArrow = makeArrow(THREE, 0xdd8b22);
      scene.add(spinArrow, bArrow);

      const rFlat = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        const v = toThree(THREE, rArr[i]);
        rFlat[3 * i] = v.x; rFlat[3 * i + 1] = v.y; rFlat[3 * i + 2] = v.z;
      }
      const trailGeom = new THREE.BufferGeometry();
      trailGeom.setAttribute("position", new THREE.BufferAttribute(rFlat, 3));
      trailGeom.setDrawRange(0, 1);
      const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x4a7ac0 }));
      scene.add(trailLine);

      function drawProb(i) {
        const ctx = probCanvas.getContext("2d");
        const W = probCanvas.width, H = probCanvas.height;
        ctx.clearRect(0, 0, W, H);
        const mL = 36, mB = 28, mT = 12, mR = 10;
        const pw = W - mL - mR, ph = H - mT - mB;
        const tMax = times[N - 1];
        const xOf = (t) => mL + (t / tMax) * pw;
        const yOf = (p) => mT + (1 - p) * ph;
        ctx.strokeStyle = "#edf0f3";
        ctx.lineWidth = 1;
        [0, 0.5, 1].forEach((p) => {
          ctx.beginPath(); ctx.moveTo(mL, yOf(p)); ctx.lineTo(W - mR, yOf(p)); ctx.stroke();
        });
        ctx.strokeStyle = "#999"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mL, mT); ctx.lineTo(mL, H - mB); ctx.lineTo(W - mR, H - mB);
        ctx.stroke();
        ctx.fillStyle = "#243247"; ctx.font = "10px Georgia, serif"; ctx.textAlign = "right";
        [0, 0.5, 1].forEach((p) => ctx.fillText(p.toFixed(1), mL - 5, yOf(p) + 3));
        ctx.textAlign = "center";
        ctx.fillText("Time t", mL + pw / 2, H - 6);

        function line(arr, color, dash) {
          ctx.strokeStyle = color; ctx.lineWidth = 2;
          ctx.setLineDash(dash || []);
          ctx.beginPath();
          for (let k = 0; k < N; k++) {
            const x = xOf(times[k]), y = yOf(arr[k]);
            if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
          ctx.setLineDash([]);
        }
        line(pDown, "#235ba8");
        line(pX, "#b04d86", [4, 3]);

        ctx.fillStyle = "#235ba8";
        ctx.beginPath(); ctx.arc(xOf(times[i]), yOf(pDown[i]), 4, 0, 7); ctx.fill();
        ctx.fillStyle = "#b04d86";
        ctx.beginPath(); ctx.arc(xOf(times[i]), yOf(pX[i]), 4, 0, 7); ctx.fill();
      }

      let currentIndex = 0;
      let playing = false;
      function updateFrame(i) {
        currentIndex = i;
        updateArrow(THREE, spinArrow, toThree(THREE, rArr[i]));
        updateArrow(THREE, bArrow, toThree(THREE, bArr[i]));
        trailLine.geometry.setDrawRange(0, i + 1);
        slider.value = String(i);
        timeLabel.textContent = "t = " + times[i].toFixed(2);
        drawProb(i);
      }

      playBtn.onclick = () => { playing = true; };
      pauseBtn.onclick = () => { playing = false; };
      resetBtn.onclick = () => { playing = false; updateFrame(0); };
      slider.oninput = (e) => { playing = false; updateFrame(parseInt(e.target.value, 10)); };

      updateFrame(0);

      let lastStep = 0;
      function loop(now) {
        requestAnimationFrame(loop);
        controls.update();
        if (playing && now - lastStep > 65) {
          lastStep = now;
          const next = currentIndex + 1 >= N ? 0 : currentIndex + 1;
          updateFrame(next);
          if (next === 0) playing = false;
        }
        renderer.render(scene, camera);
      }
      requestAnimationFrame(loop);
    }

    export default { render };
    """

    class BlochWidget(anywidget.AnyWidget):
        _esm = ESM
        title = traitlets.Unicode("").tag(sync=True)
        times = traitlets.List([]).tag(sync=True)
        r = traitlets.List([]).tag(sync=True)
        b_axis = traitlets.List([]).tag(sync=True)
        p_down = traitlets.List([]).tag(sync=True)
        p_x = traitlets.List([]).tag(sync=True)

    return (BlochWidget,)


@app.cell(hide_code=True)
def _(
    BlochWidget,
    bloch_vector,
    evolve_spin,
    initial_spin,
    mo,
    np,
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
        b_axis = np.zeros((len(times), 3))
        for i, t in enumerate(times):
            v = np.array([w1, 0, w0+w]) if view == "Rotating" else np.array([w1*np.cos(w*t), w1*np.sin(w*t), w0])
            if np.linalg.norm(v) > 0:
                b_axis[i] = v/np.linalg.norm(v)

        widget = mo.ui.anywidget(BlochWidget(
            times=times.tolist(), r=r.tolist(), b_axis=b_axis.tolist(),
            p_down=p_down.tolist(), p_x=p_x.tolist(),
            title=f"Bloch sphere · {view.lower()}",
        ))

        widgets = [controls["initial"], controls["omega0"]]
        if rotating:
            widgets += [controls["omega1"], controls["omega"], controls["frame"]]
        widgets += [controls["end"], mo.accordion({"Custom initial angles":
                   mo.vstack([controls["theta"], controls["phi"],
                              mo.md("Used only with **Custom angles**.")])})]
        sidebar = mo.vstack(widgets, gap=0.3)

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
            mo.hstack([sidebar, widget], widths=[1, 4], align="start", gap=0.6),
            mo.md("Drag the sphere any time, including while playing. " + details),
        ], gap=0.3).style({"border":"1px solid #dce3eb", "border-radius":"12px", "padding":"12px", "margin-bottom":"14px"})
        return panel

    return (simulation_panel,)


if __name__ == "__main__":
    app.run()
