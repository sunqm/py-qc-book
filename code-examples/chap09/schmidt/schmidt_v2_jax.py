import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

@jax.jit
def schmidt_orth(s):
    n = s.shape[0]
    s = jnp.asarray(s)
    c0 = jnp.identity(n)
    cs = []
    for j in range(n):
        fac = s[j,j]
        vec = c0[j]
        for k in range(j):
            dot_kj = cs[k].dot(s[j])
            vec -= dot_kj * cs[k]
            fac -= dot_kj * dot_kj

        vec /= fac**.5
        cs.append(vec)
    return jnp.asarray(cs).T

if __name__ == '__main__':
    s = np.random.rand(20, 20)
    s = s.dot(s.T)
    cs = schmidt_orth(s)
    ref = np.linalg.inv(np.linalg.cholesky(s))
    print(abs(cs - ref.T).max())
