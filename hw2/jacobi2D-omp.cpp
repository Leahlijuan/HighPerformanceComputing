#include <stdio.h>
#include "utils.h"
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

double norm(double **val, long N)
{
    double s = 0.0;
#pragma omp parallel for reduction(+ \
                                   : s)
    for (long i = 1; i <= N; i++)
    {
        for (long j = 1; j <= N; j++)
        {
            s += val[i][j] * val[i][j];
        }
    }

    return sqrt(s);
}

int main(int argc, char **argv)
{
    long N = read_option<long>("-n", argc, argv);
    long K = 5000;
    Timer t;
    t.tic();
    // memory allocation
    double **u = (double **)malloc(sizeof(double *) * (N + 2));
    double **new_u = (double **)malloc(sizeof(double *) * (N + 2));
    double **f = (double **)malloc(sizeof(double *) * (N + 2));
    double **a = (double **)malloc(sizeof(double *) * (N + 2));
    double **val = (double **)malloc(sizeof(double *) * (N + 2));
    double h = 1.0 / (N + 1);
    double h2 = h * h;
#pragma omp parallel for
    for (long i = 0; i <= N + 1; i++)
    {
        u[i] = (double *)malloc(sizeof(double) * (N + 2));
        new_u[i] = (double *)malloc(sizeof(double) * (N + 2));
        f[i] = (double *)malloc(sizeof(double) * (N + 2));
        val[i] = (double *)malloc(sizeof(double) * (N + 2));
        a[i] = (double *)malloc(sizeof(double) * (N + 2));
    }
// initialize
#pragma omp parallel for collapse(2)
    for (long i = 0; i <= N + 1; i++)
    {
        for (long j = 0; j <= N + 1; j++)
        {
            u[i][j] = 0.0;
            new_u[i][j] = 0.0;
            f[i][j] = 1.0;
            val[i][j] = 1.0;
            if (i == j)
            {
                a[i][j] = 2.0 / h2;
            }
            else if (i - j == 1 || j - i == 1)
            {
                a[i][j] = -1.0 / h2;
            }
            else
            {
                a[i][j] = 0.0;
            }
        }
    }
    double initial_residual = norm(f, N);

    // iterations
    for (long iteration = 1; iteration <= K; iteration++)
    {
#pragma omp parallel for collapse(2)
        for (long i = 1; i <= N; i++)
        {
            for (long j = 1; j <= N; j++)
            {
                new_u[i][j] = 0.25 * (h2 * f[i][j] + u[i - 1][j] + u[i][j - 1] + u[i + 1][j] + u[i][j + 1]);
            }
        }
        u = new_u;
// compute ||Au - f||
#pragma omp parallel for collapse(2)
        for (long i = 1; i <= N; i++)
        {
            for (long j = 1; j <= N; j++)
            {
                double s = 0.0;
                for (long p = 1; p <= N; p++)
                {
                    s += a[i][p] * u[p][j];
                }
                val[i][j] = s - f[i][j];
            }
        }
        if (iteration % 100 == 0)
        {
            double norm_val = norm(val, N);
            printf("iteration: %ld, the result norm is %f\n", iteration, norm_val);
        }
    }
#pragma omp parallel for
    for (int i = 0; i <= N + 1; i++)
    {
        free(u[i]);
        // free(new_u[i]);
        free(f[i]);
        free(val[i]);
    }
    free(u);
    // free(new_u);
    free(a);
    free(val);
    double time = t.toc();
    printf("N: %ld, runtime: %f\n", N, time);

    return 0;
}
