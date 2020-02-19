/*
 * Cell.h
 *
 *  Created on: Jul 21, 2012
 *      Author: barry
 */

#ifndef CELL_H_
#define CELL_H_

#include "Segment.h"
#include "SegmentUpdateInfo.h"

typedef struct CellType {
  struct RegionType* region;
  struct ColumnType* column;
  int index;
  int id;

  /**
   * The predictionSteps is the fewest number of time steps until this Cell
   * believes it will becomes active. The last prediction steps value
   * represents the fewest number of time steps this Cell believes it will
   * becomes active in.  This value will often be a count down that approaches
   * zero as time steps move forward and the Cell gets closer to becoming
   * activated.  If the Cell is not currently in a predicting state this value
   * should be ignored.
   */
  int predictionSteps;

  bool isActive;
  bool wasActive;
  bool isPredicting;
  bool wasPredicted;
  bool isLearning;
  bool wasLearning;

  Segment* segments;
  int numSegments;
  int allocatedSegments;

  SegmentUpdateInfo* segmentUpdates;
  int numSegUpdates;
  int allocatedSegUpdates;

} Cell;

__device__ void initCell(Cell* cell, struct ColumnType* column, int index);
__device__ void initInputCell(Cell* cell, struct RegionType* region, int index);
__device__ void deleteCell(Cell* cell);
__device__ void setCellPredicting(Cell* cell, bool predicting);
__device__ int numCellSegments(Cell* cell, int predictionSteps);
__device__ void nextCellTimeStep(Cell* cell);
__device__ Segment* createCellSegment(Cell* cell);
__device__ Segment* getPreviousActiveSegment(Cell* cell);
__device__ SegmentUpdateInfo* updateSegmentActiveSynapses(Cell* cell, bool previous, int segmentID, bool newSynapses);
__device__ void applyCellSegmentUpdates(Cell* cell, bool positiveReinforcement);
__device__ Segment* getBestMatchingPreviousSegment(Cell* cell, int* segmentID);
__device__ Segment* getBestMatchingSegment(Cell* cell, int numPredictionSteps,
    bool previous, int* segmentID);

#endif /* CELL_H_ */
