import asyncio
import numpy as np

async def compute_eri(k, n):
    # Mimic the function to compute ERIs
    np.random.seed(k)
    return np.random.rand(n,n,n)

async def get_j_task(i, dm):
    n = dm.shape[0]
    eri = await compute_eri(i, n)
    await asyncio.sleep(0.5)
    return np.einsum('jkl,j->kl', eri, dm[:,i])

async def get_j(dm):
    n = dm.shape[0]
    output = np.zeros((n,n))
    fs = [asyncio.ensure_future(get_j_task(i, dm)) for i in range(n)]
    for fut in fs:
        output += await fut
    return output

if __name__ == '__main__':
    n = 200
    dm = np.identity(n)
    co = get_j(dm)
    output = asyncio.get_event_loop().run_until_complete(co)
    print(output.sum())
