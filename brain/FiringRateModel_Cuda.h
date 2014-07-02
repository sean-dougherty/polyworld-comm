#pragma once

#include "FiringRateModel_Common.h"

struct FiringRateModel_Cuda {

    struct Neuron {
        float bias;
        float tau;
    };
    
    struct Synapse {
        short fromneuron;
        unsigned short partition;
        float efficacy;
        float lrate;
    };

    struct NeuronActivationPartition {
        short toneuron;
        short offset;
        short len;
    };

    FiringRateModel_Cuda();
    ~FiringRateModel_Cuda();

    void init(FiringRateModel__Neuron *neurons,
              short neurons_count, short input_neurons_count,
              FiringRateModel__Synapse *synapses,
              long synapses_count,
              float logistic_slope);

    void update(float *neuronactivation, float *newneuronactivation);

    struct GpuState {
        short neurons_count;
        short input_neurons_count;
        unsigned short partitions_count;
        long synapses_count;
        float logistic_slope;
        

        struct {
            Neuron *neurons;
            Synapse *synapses;
            NeuronActivationPartition *partitions;
            
            float *neuronactivation;
            float *newneuronactivation;
        } buffers;
    } gpu;
};
