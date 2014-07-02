#pragma once

#include "FiringRateModel_Common.h"

struct FiringRateModel_Cuda {
    
    struct NeuronSynapsesPartition {
        short toneuron;
        short offset;
        short len;
    };
    
    struct SynapseEndpoints {
        short toneuron;
        short fromneuron;
    };

    FiringRateModel_Cuda();
    ~FiringRateModel_Cuda();

    void init(FiringRateModel__Neuron *neurons,
              short neurons_count, short input_neurons_count,
              FiringRateModel__Synapse *synapses,
              long synapses_count);

    void update(float *neuronactivation, float *newneuronactivation);

    struct GpuState {
        short neurons_count;
        short input_neurons_count;
        long synapses_count;

        struct {
            float *bias;

            NeuronSynapsesPartition *partitions;
            SynapseEndpoints *endpoints;
            float *efficacy;
            float *lrate;
            long *partition_index;
            float *neuronactivation;
            float *newneuronactivation;
        } buffers;
    } gpu;
};
