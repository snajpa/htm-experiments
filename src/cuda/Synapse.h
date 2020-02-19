/*
 * Synapse.h
 *
 *  Created on: Jul 19, 2012
 *      Author: barry
 */

#ifndef SYNAPSE_H_
#define SYNAPSE_H_

/* Global parameters that apply to all Region instances */
#define MAX_PERM 10000 /* Maximum/full permanence value */
#define CONNECTED_PERM 2000 /* Synapses with permanences above this value are connected. */
#define INITIAL_PERMANENCE 3000 /*initial permanence for distal synapses*/
#define PERMANENCE_INC 150 /*Amount permanences of synapses are incremented in learning*/
#define PERMANENCE_DEC 100 /*Amount permanences of synapses are decremented in learning*/

/*
 * A data structure representing a synapse. Contains a permanence value and the
 * source input index.  Also contains a 'location' in the input space that this
 * synapse roughly represents.
 */
typedef struct SynapseType {
  struct CellType* inputSource;
  int permanence;
  bool isConnected;
  bool wasConnected;
} Synapse;

__device__ void initSynapse(Synapse* syn, struct CellType* inputSource, int permanence);
/*__device__ bool isSynapseConnected(Synapse* syn);*/
__device__ bool isSynapseActive(Synapse* syn, bool connectedOnly);
__device__ bool wasSynapseActive(Synapse* syn, bool connectedOnly);
__device__ bool wasSynapseActiveFromLearning(Synapse* syn);
__device__ void increaseSynapsePermanence(Synapse* syn, int amount);
__device__ void decreaseSynapsePermanence(Synapse* syn, int amount);

#endif /* SYNAPSE_H_ */
