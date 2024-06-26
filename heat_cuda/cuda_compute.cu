#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include "compute.h"
#include "output.h"


/* A 2d heat dissipation algorithm */


__constant__ double wo = 0.14644660940672627; // Weight of Orthogonal neighbours
__constant__ double wd = 0.10355339059327377; // Weight of Diagonal neighbours

void reduce(const struct parameters* p, struct results *r, int M, int N,
            double *t1, double *t2, int iter) {
    // Gather results
    double tmin = t1[(M+2) + 1];
    double tmax = t1[(M+2) + 1];
    double tavg = 0;
    double maxdiff = 0;
    for (int j = 1; j < N+1; j++) {
        for (int i = 1; i < M+1; i++) {
            double t = t1[j * (M+2) + i];
            double told = t2[j * (M+2) + i];
            if (t < tmin) {
                tmin = t;
            }
            if (t > tmax) {
                tmax = t;
            }
            tavg += t/(N*M);
            double diff = told - t;
            diff = diff < 0 ? -diff : diff;
            if (diff > 1) {
              printf("diff is %f at %d, %d\n", diff, i, j);
            }
            if (diff > maxdiff) maxdiff = diff;
        }
    }
    r->niter = iter;
    r->tmin = tmin;
    r->tmax = tmax;
    r->tavg = tavg;
    r->maxdiff = maxdiff;
}

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda error \n");
        exit(1);
    }
}

/* Main heat dissipation kernel */
__global__ void heatKernel(double* t1, double* t2, double* c1, int n, size_t M, size_t N) {
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        const size_t i = k%M + 1;
        const size_t j = k/M + 1;
        const size_t M2 = M+2;
        const size_t jM2 = j*M2;
        const double c = c1[jM2 + i];

        t2[jM2 + i] = c * t1[jM2 + i] + (1 - c) * (wo * (
                    t1[jM2 + i-1]
                + t1[jM2 + i+1]
                + t1[jM2 - M2 + i]
                + t1[jM2 + M2 + i])
                + wd * (
                    t1[jM2 - M2 + i-1]
                + t1[jM2 - M2 + i+1]
                + t1[jM2 + M2 + i-1]
                + t1[jM2 + M2 + i+1]));
    }
}

/* Kernel to copy the halo cells */
__global__ void haloKernel(double* t1, double* t2, int n, size_t M) {
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t M2 = M+2;
    if (j < n) {
        t1[j * M2] = t1[j * M2 + M];
    } else if (j < (n*2)){
        j -= n;
        t1[j * M2 + (M+1)] = t1[j * M2 + 1];
    }
}

extern "C" 
void cuda_do_compute(const struct parameters* p, struct results *r) {
    const size_t M = p->M;
    const size_t N = p->N;
    size_t size = (N+2) * (M+2) * sizeof(double);
    double *t1 = (double*)malloc(size);
    double *t2 = (double*)malloc(size);
    double *c1 = (double*)malloc(size);
    double *swp;
    
    // Copy input array into working array
    for (int i = 1; i < M+1; i++) {
        for (int j = 1; j < N+1; j++) {
            t1[j * (M+2) + i] = p->tinit[(j-1)*M + i-1];
        }
    }

    /* Conductivity array with same indexing for convenience,
     * beware edges are uninitialized */
    for (int j = 1; j < N+1; j++) {
        for (int i = 1; i < M+1; i++) {
            c1[j * (M+2) + i] = p->conductivity[(j-1)*M + i-1];
        }
    }

    // init top/bottom halo cells
    for (int i = 1; i < M+1; i++) {
        t1[i] = t2[i] = t1[(M+2) + i];
        t1[(N+1) * (M+2) + i] = t2[(N+1) * (M+2) + i] = t1[N * (M+2) + i];
    }

    const int maxit = p->maxiter;
    int iter;

    // do cuda allocs and copys
    double *device_t1 = NULL;
    checkCudaCall(cudaMalloc((void **) &device_t1, size));
    if (device_t1 == NULL) {
        printf("Error in cudaMalloc! \n");
        return;
    }
    double *device_t2 = NULL;
    checkCudaCall(cudaMalloc((void **) &device_t2, size));
    if (device_t2 == NULL) {
        printf("Error in cudaMalloc! \n");
        return;
    }
    double *device_c1 = NULL;
    checkCudaCall(cudaMalloc((void **) &device_c1, size));
    if (device_c1 == NULL) {
        printf("Error in cudaMalloc! \n");
        return;
    }
    checkCudaCall(cudaMemcpy(device_t1, t1, size, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(device_t2, t2, size, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(device_c1, c1, size, cudaMemcpyHostToDevice));

    int blockSize = 128; // Empirically determined
    int n = M*N;
    int halo_n = N+2;

    for (iter = 0; iter < maxit; iter++) {
        // swap halo cells
        haloKernel<<<((halo_n*2)+blockSize-1)/blockSize, blockSize>>>(device_t1, device_t2, halo_n, M);

        // execute kernel
        heatKernel<<<(n+blockSize-1)/blockSize, blockSize>>>(device_t1, device_t2, device_c1, n, M, N);

        // swap ptrs
        swp = device_t1;
        device_t1 = device_t2;
        device_t2 = swp;
    }
    // Wait for last kernel and check errors
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    // get results
    checkCudaCall(cudaMemcpy(t1, device_t1, size, cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(t2, device_t2, size, cudaMemcpyDeviceToHost));

    reduce(p, r, M, N, t1, t2, iter);

    checkCudaCall(cudaFree(device_t1));
    checkCudaCall(cudaFree(device_t2));
    checkCudaCall(cudaFree(device_c1));
}
