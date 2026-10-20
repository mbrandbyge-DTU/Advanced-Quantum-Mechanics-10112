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
