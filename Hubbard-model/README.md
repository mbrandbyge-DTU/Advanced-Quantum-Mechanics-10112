## The Hubbard Model for two electrons on a ring

This problem studies the Hubbard model in one dimension.
The Hubbard model approximates the long-range Coulomb repulsion between 
electrons as an on-site repulsion energy $U.$  The model is otherwise
a tight-binding Hamiltonian with nearest-neighbor hopping.

The Hamiltonian is written
$$H=H_{TB}+H_C=-\sum_{n,s}t(c^\dagger_{n+1,s}c_{n,s}
	  	      + c^\dagger_{n,s}c_{n+1,s}) +
\sum_{n} U c^\dagger_{n,\uparrow}c_{n,\uparrow}
	c^\dagger_{n,\downarrow}c_{n,\downarrow}$$

$n$ is running from $1$ to $L$, where we imagine the L sites organized in
a ring so that the electron can also hop from site $L$ to site $1$.
We can see that $t$, the hopping matrix element, represents a kinetic 
energy for the electrons which is lowered if the wavefunction delocalizes
into long-wavelength plane waves.  

## Open and experiment

### Interactive notebook (recommended)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/mbrandbyge-DTU/Advanced-Quantum-Mechanics-10112/blob/main/Hubbard-model/hubbard_ring.py/wasm)

`hubbard_ring.py` contains two experiments, each with its own controls:

1. **Spectrum and eigenstates.** Energies versus $U/t$, split into singlets and triplets.
   Choose an eigenstate and see its pair amplitude $\phi(k)$ and the separation
   probability $P(r)$, where $r=n_1-n_2$.
2. **The correlation hole.** Fix the ↑ electron and see where the ↓ electron is on the ring.
   The plot shows the double occupancy $P(0)$ versus $U$, together with the slope
   $dE_0/dU$ (Hellmann–Feynman).

The physics is in three short NumPy cells. The main one is `pair_hamiltonian`, which
students can edit, for example to add a nearest-neighbour repulsion.
The notebook ends with predict → try → explain questions and collapsed discussion notes.

The link runs entirely in the browser; no installation or account is needed. For local use:

```bash
marimo edit --sandbox hubbard_ring.py
```

### Plain Jupyter notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbrandbyge-DTU/Advanced-Quantum-Mechanics-10112/blob/main/Hubbard-model/Hubbard_on_a_ring.ipynb)

`Hubbard_on_a_ring.ipynb` is the original, minimal notebook for students who want to write
the numerical solution themselves, step by step.
