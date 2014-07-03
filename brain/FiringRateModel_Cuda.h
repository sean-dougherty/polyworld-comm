#pragma once

#define EXEC_CPU false

#include "FiringRateModel_Common.h"

struct FiringRateModel_Cuda {

    struct AgentState {
        FiringRateModel_Cuda *model;
        float *neuronactivation;
        float *newneuronactivation;
    };

    static void update(AgentState *agents, long nagents);

    struct Neuron {
        float bias;
        float tau;
    };
    
    struct Synapse {
        short fromneuron;
        unsigned short partition;
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
              short neurons_count, short input_neurons_count, short output_neurons_count,
              float *neuronactivation,
              FiringRateModel__Synapse *synapses,
              long synapses_count,
              float logistic_slope,
              float decay_rate,
              float max_weight);

    void update(float *neuronactivation,
                float *newneuronactivation,
                FiringRateModel__Synapse *synapses);

    struct GpuState {
        short neurons_count;
        short input_neurons_count;
        short output_neurons_count;
        unsigned short partitions_count;
        long synapses_count;
        float logistic_slope;
        float decay_rate;
        float max_weight;

        struct {
            Neuron *neurons;
            Synapse *synapses;
            NeuronActivationPartition *partitions;
            
            float *inputactivation;
            float *neuronactivation;
            float *newneuronactivation;
            float *efficacy;
        } buffers;
    } gpu;
};
