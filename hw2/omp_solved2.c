/******************************************************************************
 * FILE: omp_bug2.c
 * DESCRIPTION:
 *   Another OpenMP program with a bug.
 * AUTHOR: Blaise Barney
 * LAST REVISED: 04/06/05
 ******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int nthreads, i;
  float total;

/*** Spawn parallel region ***/
#pragma omp parallel
  {
    /* Obtain thread number */
    const int tid = omp_get_thread_num(); // since this tid should not be shared between different threads, so it should be private
    /* Only master thread does this */
    if (tid == 0)
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
    printf("Thread %d is starting...\n", tid);

#pragma omp barrier

    /* do some work */
    total = 0.0;
#pragma omp parallel for reduction(+ \
                                   : total) // need to get sum, should do reduction
    for (i = 0; i < 1000000; i++)
      total = total + i * 1.0;

    printf("Thread %d is done! Total= %e\n", tid, total);

  } /*** End of parallel region ***/
}
