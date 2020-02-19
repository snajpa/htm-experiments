/*
 * Region.h
 *
 *  Created on: Jul 21, 2012
 *      Author: barry
 */

#ifndef REGION_H_
#define REGION_H_

#include "Column.h"

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

typedef struct RegionType {
  int inputWidth, inputHeight;
  int localityRadius;
  int cellsPerCol;
  int segActiveThreshold;
  int newSynapseCount;

  float pctInputPerCol;
  float pctMinOverlap;
  float pctLocalActivity;

  bool spatialHardcoded;
  bool spatialLearning;
  bool temporalLearning;

  int width, height;
  float xSpace, ySpace;

  Column* columns;
  int numCols;

  float minOverlap;
  float inhibitionRadius;
  int desiredLocalActivity;

  char* inputData;
  int nInput;
  Cell* inputCells;
  int iters;
} Region;

__global__ void newRegionHardcoded(Region *region, int inputSizeX, int inputSizeY, int localityRadius,
    int cellsPerCol, int segActiveThreshold, int newSynapseCount,
    char* inputData);
__global__ void newRegion(Region *region, int inputSizeX, int inputSizeY, int colGridSizeX, int colGridSizeY,
    float pctInputPerCol, float pctMinOverlap, int localityRadius,
    float pctLocalActivity, int cellsPerCol, int segActiveThreshold,
    int newSynapseCount, char* inputData);
__global__ void deleteRegion(Region* region);
__device__ void getColumnPredictions(Region* region, char* outData);
__device__ void getLastAccuracy(Region* region, float* result);
__device__ int numRegionActiveColumns(Region* region);
__global__ void numRegionSegments(Region* region, int predictionSteps, int* ret);
__global__ void runOnce(Region* region);

__device__ void performSpatialPooling(Region* region);
__device__ void performTemporalPooling(Region* region);

#endif /* REGION_H_ */
