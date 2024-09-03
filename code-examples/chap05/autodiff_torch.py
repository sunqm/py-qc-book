import numpy as np
import torch

def coulomb_energy(coords, z):
    zz = z[:,None] * z
    d = torch.linalg.norm(coords[:,None,:] - coords, axis=2)
    tril = np.tril_indices(len(z), -1)
    return (zz[tril] / d[tril]).sum()

class CoulombEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, z):
        zz = z[:,None] * z
        rr = coords[:,None,:] - coords
        d = np.linalg.norm(coords[:,None,:] - coords, axis=2)
        tril = np.tril_indices(len(z), -1)
        res = (zz[tril] / d[tril]).sum()
        # ctx.save_for_backward can cache information in ctx for backward computation.
        ctx.save_for_backward(coords, z)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        coords, z = ctx.saved_tensors
        zz = z[:,None] * z
        rr = coords[:,None,:] - coords
        d = np.linalg.norm(coords[:,None,:] - coords, axis=2)
        with np.errstate(all='ignore'):
            rinv = 1. / d
            rinv[np.diag_indices(len(z))] = 0.
        r_grad = -np.einsum('i,j,ijx,ij->ix', z, z, rr, rinv**3)
        z_grad = np.einsum('j,ij->i', z, rinv)
        # grad_output represents the factor associated with the current function.
        return grad_output * r_grad, grad_output * z_grad


if __name__ == '__main__':
    n = 2
    coords = torch.rand(n, 3, requires_grad=True)
    Z = torch.rand(n, requires_grad=True)
    e = coulomb_energy(coords, Z)
    print(e.backward())
    print(coords.grad)
    print(Z.grad)

    coords.grad = None
    Z.grad = None
    e = CoulombEnergy.apply(coords, Z)
    e.backward()
    print(coords.grad)
    print(Z.grad)
