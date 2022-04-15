#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <stdlib.h>

double sum_of_neighbors(int N, double *u, int i, int j)
{
    double val_left = 0.0, val_right = 0.0, val_up = 0.0, val_down = 0.0;

    if (i > 0)
    {
        val_up = u[(i - 1) * N + j];
    }
    if (i < N - 1)
    {
        val_down = u[(i + 1) * N + j];
    }
    if (j > 0)
    {
        val_left = u[i * N + (j - 1)];
    }
    if (j < N - 1)
    {
        val_right = u[i * N + (j + 1)];
    }

    return val_left + val_right + val_up + val_down;
}

void A_times(int N, double *u, double *ret)
{
    double h_sq = 1.0 / ((N + 1) * (N + 1));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int flat_ix = i * N + j;

            double val_here = u[flat_ix];

            ret[flat_ix] =
                1.0 / h_sq *
                (4.0 * val_here - sum_of_neighbors(N, u, i, j));
        }
    }
}

double residual(int N, double *u, double *f)
{
    double *Au = (double *)malloc(N * N * sizeof(double));
    A_times(N, u, Au);
    double residual_norm_sq = 0.0;
    for (int i = 0; i < N * N; i++)
    {
        residual_norm_sq += (Au[i] - f[i]) * (Au[i] - f[i]);
    }

    free(Au);

    return sqrt(residual_norm_sq);
}

void jacobi_iters_par(int N, int iters, double *f, double *u_prev, double *u)
{
    double h_sq = 1.0 / ((N + 1) * (N + 1));

    for (int ix = 0; ix < iters; ix++)
    {

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int flat_ix = i * N + j;
                u[flat_ix] = 0.25 *
                             (h_sq * f[flat_ix] + sum_of_neighbors(N, u_prev, i, j));
            }
        }

        // Swap u and u_prev
        double *tmp = u;
        u = u_prev;
        u_prev = tmp;
    }

    // swap
    double *tmp = u;
    u = u_prev;
    u_prev = tmp;
}

#define BLOCK_SIZE 32

__global__ void jacobi_kernel(double *u, const double *f, const double *u_prev, double h_sq, long N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double val_left = 0.0, val_right = 0.0, val_up = 0.0, val_down = 0.0;

    if (i < N && j < N)
    {
        if (i > 0)
        {
            val_up = u_prev[(i - 1) * N + j];
        }
        if (i < N - 1)
        {
            val_down = u_prev[(i + 1) * N + j];
        }
        if (j > 0)
        {
            val_left = u_prev[i * N + (j - 1)];
        }
        if (j < N - 1)
        {
            val_right = u_prev[i * N + (j + 1)];
        }

        u[N * i + j] = 0.25 * (h_sq * f[N * i + j] + val_up + val_down + val_left + val_right);
    }
}

void jacobi_wrapper(double *u, const double *f, long N, long iters)
{
    double *u_d, *f_d, *u_prev_d;

    cudaMalloc(&u_d, N * N * sizeof(double));
    cudaMalloc(&f_d, N * N * sizeof(double));
    cudaMalloc(&u_prev_d, N * N * sizeof(double));

    // Initialize both current and previous step to input
    cudaMemcpyAsync(u_d, u, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(u_prev_d, u, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(f_d, f, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    double h_sq = 1.0 / ((N + 1) * (N + 1));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE + 1, N / BLOCK_SIZE + 1);

    // Iterate kernel, copying current step to previous each time
    for (int k = 0; k < iters; k++)
    {
        jacobi_kernel<<<dimGrid, dimBlock>>>(u_d, f_d, u_prev_d, h_sq, N);
        cudaMemcpyAsync(u_prev_d, u_d, N * N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    // Return final result to host memory
    cudaMemcpyAsync(u, u_prev_d, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(u_d);
    cudaFree(f_d);
    cudaFree(u_prev_d);
}

int main(int argc, char **argv)
{
    // General setup
    long N = read_option<long>("-n", argc, argv);
    long iters = 5000;

    Timer timer;

    double *f = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++)
    {
        f[i] = 1.0;
    }

    // CPU calculation
    double *u_prev = (double *)malloc(N * N * sizeof(double));
    double *u = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++)
    {
        u_prev[i] = 0.0;
        u[i] = 0.0;
    }

    printf("CPU:\n");
    timer.tic();
    jacobi_iters_par(N, iters, f, u, u_prev);

    double time = timer.toc();

    printf("residual:   %3f\n", residual(N, u, f));
    printf("time:   %3f\n\n", time);

    free(u_prev);
    free(u);

    // GPU calculation
    cudaMallocHost(&u, N * N * sizeof(double));
    for (int i = 0; i < N * N; i++)
    {
        u[i] = 0.0;
    }

    printf("GPU:\n");

    timer.tic();
    jacobi_wrapper(u, f, N, iters);
    time = timer.toc();

    printf("residual:   %3f\n", residual(N, u, f));
    printf("time:   %3f\n\n", time);

    free(f);
    cudaFree(u);

    return 0;
}