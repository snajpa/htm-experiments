/*
 * Main file containing a set of tests related to the HTM code.
 * A few tests are standard unit tests for functionality of the
 * HTM Region and its components.  Few others are performance
 * tests for HTM.  Finally there are some miscellaneous tests
 * related to lower C level code used within the HTM implementation.
 *
 * The tests currently simply print to standard out any failures that
 * are encountered.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Region.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
return EXIT_FAILURE;}} while(0)

#define MAX_FILE_SIZE (0x100000)
#define DEBUG 1

/**
 * This test creates a hardcoded Region of size 25x25 (625) and runs
 * 10,000 iterations of the Region using randomly generated data
 * (40 active columns out of 625).  The nunique parameter defines
 * how many unique data configurations to randomly choose from between
 * iterations.  If zero passed in use true random data configurations
 * (no unique limit).  The performance stats are printed to standard
 * out every 1000 iterations.
 */
void testRegionPerformance(unsigned int nunique) {
  printf("testRegionPerformance(%i)...\n", nunique);

  int nx = 25;
  int ny = 25;
  int localityRadius = 0;
  int cellsPerCol = 4;
  int segActiveThreshold = 3;
  int newSynapseCount = 5;

  float acc[2];
  int* scn = cudaMalloc(sizeof(int));
  char* data = cudaMalloc(nx*ny * sizeof(char));
  Region* region = cudaMalloc(sizeof(Region));
  region = newRegionHardcoded(nx,ny, localityRadius, cellsPerCol,
      segActiveThreshold, newSynapseCount, data);

  /*create a sequence of length 10.  repeat it 10 times and check region accuracy. */
  int niters = 10000;
  int nactive = 40;
  int ncol = nx*ny;
  unsigned long si = 1;
  srand(42);

  unsigned long time = clock();
  double otime = 0;/*omp_get_wtime();*/

  int i,j,r;
  for(i=0; i<=niters; ++i) {
    /*select a random 40 data positions*/
    /*choose nact random column indicies to represent word*/
    for(j=0; j<ncol; ++j)
      data[j] = 0;

    /*if non-zero, reseed the rand using one of nunique seed values.*/
    if(nunique > 0) {
      srand(si*4101);
      si *= 5303;
      srand(rand() % nunique);
    }

    r=0;
    while(r < nactive) {
      int d = rand() % ncol;
      if(data[d]==0) {
        data[d] = 1;
        r++;
      }
    }

    runOnce(region);

    if(i % 1000 == 0) {
      unsigned long elapse = clock() - time;
      double oelapse = 1;/*omp_get_wtime() - otime;*/
      printf("iters %i: time %f (%lu)\n", i, oelapse, elapse/1000);

      /*print how many segments of particular counts that exist*/
      int sn;
      for(sn=0; sn<12; ++sn) {
        numRegionSegments(region, sn, scn);
        printf("%i(%i)  ", sn, scn);
      }
      printf("\n");

      /*getLastAccuracy(region, acc);
      printf("acc  %f   %f\n", acc[0], acc[1]);*/

      time = clock();
      /*otime = omp_get_wtime();*/
    }
  }

  unsigned long elapse = clock() - time;
  printf("iters %i: time: %lu\n", niters, elapse/1000);

  deleteRegion(region);
  cudaFree(region);
  cudaFree(data);
  cudaFree(scn);

  printf("OK\n");
}

int main(void) {
  testRegionPerformance(10);
  /*testRegionPerformance(0);*/
  return EXIT_SUCCESS;
}
