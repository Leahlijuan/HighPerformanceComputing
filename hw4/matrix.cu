#include <algorithm>
#include <stdio.h>
#include "utils.h"

#define BLOCK_SIZE 1024

void Check_CUDA_Error(const char *message)
{
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
    exit(-1);
  }
}


void matrix_vector_ref(double *Ax_ref, const double *A, const double *x, long N)
{
  for (long i = 0; i < N; i++)
  {
    for (long j = 0; j < N; j++)
    {
      Ax_ref[i] += A[N * i + j] * x[j];
    }
  }
}

__global__ void matrix_vector_kernel(double *Ax, const double *A, const double *x, long N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N)
  {
    Ax[idx] = 0;
    for (long j = 0; j < N; j++)
    {
      Ax[idx] += A[idx * N + j] * x[j];
    }
  }
}

int main(int argc, char **argv)
{
  long N = read_option<long>("-n", argc, argv);

  double *A, *x, *A_d, *x_d, *Ax_d, *Ax_ref, *Ax;

  // Initialize vector and matrix
  cudaMallocHost((void **)&A, N * N * sizeof(double));
  cudaMallocHost((void **)&x, N * sizeof(double));

  for (long i = 0; i < N; i++)
  {
    x[i] = drand48();
  }
  for (long i = 0; i < N * N; i++)
  {
    A[i] = drand48();
  }

  // Get reference product
  cudaMallocHost((void **)&Ax_ref, N * sizeof(double));
  Timer t;
  t.tic();
  matrix_vector_ref(Ax_ref, A, x, N);
  double time = t.toc();
  printf("CPU Bandwidth = %f GB/s\n", 2 * N * N * sizeof(double) / time / 1e9);

  // Get GPU product
  cudaMalloc(&A_d, N * N * sizeof(double));
  cudaMalloc(&x_d, N * sizeof(double));
  cudaMalloc(&Ax_d, N * sizeof(double));
  cudaMallocHost(&Ax, N * sizeof(double));

  cudaMemcpyAsync(A_d, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  Timer t2;
  t2.tic();
  matrix_vector_kernel<<<N / BLOCK_SIZE + 1, BLOCK_SIZE>>>(Ax_d, A_d, x_d, N);
  cudaMemcpyAsync(Ax, Ax_d, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double time2 = t2.toc();
  printf("GPU Bandwidth = %f GB/s\n", 2 * N * N * sizeof(double) / time2 / 1e9);

  double err = 0;
  for (long i = 0; i < N; i++)
  {
    err += (Ax_ref[i] - Ax[i]) * (Ax_ref[i] - Ax[i]);
  }
  printf("Error = %f\n", err);

  // Cleanup
  cudaFree(A_d);
  cudaFree(x_d);
  cudaFree(Ax_d);
  cudaFreeHost(A);
  cudaFreeHost(x);
  cudaFreeHost(Ax);
  cudaFreeHost(Ax_ref);

  return 0;
}