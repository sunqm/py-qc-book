import numpy as np
import jax.numpy as jnp

def coulomb_energy(coords, z):
    zz = z[:,None] * z
    rr = coords[:,None,:] - coords
    # Note zeros are not differentiable for norm.
    # They are discarded before calling norm.
    tril = np.tril_indices(len(z), -1)
    d = jnp.linalg.norm(rr[tril], axis=1)
    return (zz[tril] / d).sum()

if __name__ == '__main__':
    coords = jnp.array(np.random.rand(2, 3))
    Z = jnp.array(np.random.rand(2))
    print(jax.grad(coulomb_energy, argnums=(0, 1))(coords, Z))
