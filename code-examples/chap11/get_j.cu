#include <stdio.h>
#include <cuda_runtime.h>

__global__
void _kernel_v1(double *output, double *eri, double *dm, int n)
{
    size_t ij = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nn = n * n;
    if (ij > nn) return;
    size_t i = ij / n;
    size_t j = ij % n;
    size_t kl;
    double dm_ji = dm[j*n+i];
    for (kl = 0; kl < nn; ++kl) {
        atomicAdd(&output[kl], eri[(i*n+j)*nn+kl] * dm_ji);
    }
}

__global__
void _kernel_v2(double *output, double *eri, double *dm, int n)
{
    size_t kl = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nn = n * n;
    if (kl > nn) return;
    size_t i, j;
    double s = 0.;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            s += eri[(i*n+j)*nn+kl] * dm[j*n+i];
        }
    }
    output[kl] = s;
}

#define BLOCK   16
__global__
void _kernel_v3(double *output, double *eri, double *dm, int n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    size_t kl = blockIdx.x * blockDim.x*blockDim.y + ty * blockDim.x + tx;
    size_t nn = n * n;
    size_t i, j, i0, j0;
    __shared__ double dm_t[BLOCK][BLOCK];

    double s = 0.;
    for (i0 = 0; i0 < n - (BLOCK-1); i0 += BLOCK) {
        for (j0 = 0; j0 < n - (BLOCK-1); j0 += BLOCK) {
            __syncthreads();
            dm_t[tx][ty] = dm[(j0+ty)*n+i0+tx];
            __syncthreads();
            if (kl < nn) {
                for (i = 0; i < BLOCK; ++i) {
                    for (j = 0; j < BLOCK; ++j) {
                        s += eri[((i0+i)*n+j0+j)*nn+kl] * dm_t[i][j];
                    }
                }
            }
        }
    }
    if (kl < nn) {
        if (j0 < n) {
            for (i = 0; i < i0; ++i) {
                for (j = j0; j < n; ++j) {
                    s += eri[(i*n+j)*nn+kl] * dm[j*n+i];
                }
            }
        }
        for (i = i0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                s += eri[(i*n+j)*nn+kl] * dm[j*n+i];
            }
        }
        output[kl] = s;
    }
}

extern "C" {
void kernel_v1(double *output, double *eri, double *dm, int n)
{
    int blocks = (n*n+255)/256;
    int threads = 256;
    _kernel_v1<<<blocks, threads>>>(output, eri, dm, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
}

void kernel_v2(double *output, double *eri, double *dm, int n)
{
    int blocks = (n*n+255)/256;
    int threads = 256;
    _kernel_v2<<<blocks, threads>>>(output, eri, dm, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
}

void kernel_v3(double *output, double *eri, double *dm, int n)
{
    int blocks = (n*n+255)/256;
    dim3 threads(BLOCK, BLOCK, 1);
    _kernel_v3<<<blocks, threads>>>(output, eri, dm, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
}

void kernel_v1_mapped(double *output, double *eri_cpu, double *dm, int n)
{
    double *eri_gpu;
    // enable zero-copy
    cudaError_t err = cudaHostGetDevicePointer(&eri_gpu, eri_cpu, 0);
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
    kernel_v1(output, eri_gpu, dm, n);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
}

void kernel_v2_mapped(double *output, double *eri_cpu, double *dm, int n)
{
    double *eri_gpu;
    // enable zero-copy
    cudaError_t err = cudaHostGetDevicePointer(&eri_gpu, eri_cpu, 0);
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
    kernel_v2(output, eri_gpu, dm, n);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
}

void kernel_v3_mapped(double *output, double *eri_cpu, double *dm, int n)
{
    double *eri_gpu;
    // enable zero-copy
    cudaError_t err = cudaHostGetDevicePointer(&eri_gpu, eri_cpu, 0);
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
    kernel_v3(output, eri_gpu, dm, n);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err));
    }
}
}
