#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  prefix_sum[0] = 0;
  long num_threads = 0;
  long segment_length = 0;
  #pragma omp parallel shared(num_threads, segment_length)
  {
    num_threads = omp_get_num_threads();
    segment_length = (n + num_threads - 1) / num_threads;

    #pragma omp for
    for (long i = 0; i < num_threads; i++)
    {
      long segment_i = omp_get_thread_num();
      long segment_start = segment_i * segment_length;
      long segment_end = (segment_i + 1) * segment_length;
      if (segment_end > n) {
        segment_end = n;
      }

if (segment_start > 0)
    {
      prefix_sum[segment_start] = A[segment_start - 1];
    }
    else
    {
      prefix_sum[segment_start] = 0;
    }
      
for (long i = segment_start + 1; i < segment_end; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
      }  
    }
  }
   for (long segment_ix = 1; segment_ix < num_threads; segment_ix++) {
    long segment_start = segment_length * segment_ix;
    long segment_end = segment_length * (segment_ix + 1);
    if (segment_end > n) {
      segment_end = n;
    }

    for (long i = segment_start; i < segment_end; i++) {
      prefix_sum[i] += prefix_sum[segment_start - 1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
