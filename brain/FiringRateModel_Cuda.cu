#include "FiringRateModel_Cuda.h"

#include <assert.h>
#include <cuda.h>
#include <limits.h>
#include <stdio.h>
#include <iostream>

#define Threads_Per_Block 512
#define MAX_SYNAPSES_PER_THREAD 256
#define MAX_NEURONS Threads_Per_Block

#define xcuda(stmt) {                                                   \
        cudaError_t err = stmt;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << ": Failed to run " << #stmt << ". Reason: " << cudaGetErrorString(err) << std::endl; \
            assert(false);                                                    \
        }                                                               \
    }

FiringRateModel_Cuda::FiringRateModel_Cuda() {
    memset(&gpu.buffers, 0, sizeof(gpu.buffers));
}

FiringRateModel_Cuda::~FiringRateModel_Cuda() {
#define FREE(x) xcuda( cudaFree(gpu.buffers.x) );

    FREE(neurons);
    FREE(synapses);
    FREE(partitions);
    FREE(neuronactivation);
    FREE(newneuronactivation);
}

void FiringRateModel_Cuda::init(FiringRateModel__Neuron *neurons,
                                short neurons_count, short input_neurons_count,
                                FiringRateModel__Synapse *synapses,
                                long synapses_count,
                                float logistic_slope) {

    assert(neurons_count < MAX_NEURONS);

    gpu.neurons_count = neurons_count;
    gpu.input_neurons_count = input_neurons_count;
    gpu.synapses_count = synapses_count;
    gpu.logistic_slope = logistic_slope;

    int nsynapses_per_thread = (synapses_count - 1) / Threads_Per_Block + 1;
    assert(nsynapses_per_thread <= MAX_SYNAPSES_PER_THREAD);

    Neuron gpu_neurons[neurons_count];
    for(short i = 0; i < neurons_count; i++) {
        gpu_neurons[i].bias = neurons[i].bias;
        gpu_neurons[i].tau = neurons[i].tau;
    }

    NeuronActivationPartition partitions[USHRT_MAX];
    NeuronActivationPartition *currpartition = NULL;
    Synapse gpu_synapses[synapses_count];

    for(long i = 0; i < synapses_count; i++) {
        FiringRateModel__Synapse &synapse = synapses[i];
        if( (i % Threads_Per_Block == 0) || (synapse.toneuron != currpartition->toneuron) ) {
            if(currpartition)
                currpartition++;
            else
                currpartition = partitions;
            assert(currpartition - partitions < USHRT_MAX);

            currpartition->toneuron = synapse.toneuron;
            currpartition->offset = i % Threads_Per_Block;
            currpartition->len = 0;
        }
        currpartition->len++;

        Synapse &gpu_synapse = gpu_synapses[i];
        gpu_synapse.fromneuron = synapse.fromneuron;
        gpu_synapse.partition = currpartition - partitions;
        gpu_synapse.efficacy = synapse.efficacy;
        gpu_synapse.lrate = synapse.lrate;
    }

    size_t npartitions = currpartition - partitions + 1;
    gpu.partitions_count = npartitions;

/*
    for(long i = 0; i < synapses_count; i++) {
        if( (i % Threads_Per_Block) == 0 ) {
            printf("********\n");
        }
        NeuronActivationPartition &p = partitions[partition_index[i]];
        assert(p.toneuron == synapses[i].toneuron);
        printf("%5ld %3d] %3d %3d %3d\n", i, synapses[i].toneuron, p.toneuron, p.offset, p.len);
    }
    for(size_t i = 0; i < npartitions; i++) {
        NeuronActivationPartition &p = partitions[i];
        printf("%4lu] to=%3d off=%4d len=%3d\n", i, p.toneuron, p.offset, p.len);
    }
*/

    xcuda( cudaMalloc((void**)&gpu.buffers.neurons, sizeof(gpu_neurons)) );
    xcuda( cudaMemcpy(gpu.buffers.neurons, gpu_neurons, sizeof(gpu_neurons), cudaMemcpyHostToDevice) );

    xcuda( cudaMalloc((void**)&gpu.buffers.synapses, sizeof(gpu_synapses)) );
    xcuda( cudaMemcpy(gpu.buffers.synapses, gpu_synapses, sizeof(gpu_synapses), cudaMemcpyHostToDevice) );

    size_t sizeof_partitions = npartitions * sizeof(NeuronActivationPartition);
    xcuda( cudaMalloc((void**)&gpu.buffers.partitions, sizeof_partitions) );
    xcuda( cudaMemcpy(gpu.buffers.partitions, partitions, sizeof_partitions, cudaMemcpyHostToDevice) );

    size_t sizeof_activation = sizeof(float) * neurons_count;
    xcuda( cudaMalloc((void **)&gpu.buffers.neuronactivation, sizeof_activation) );
    xcuda( cudaMalloc((void **)&gpu.buffers.newneuronactivation, sizeof_activation) );
}

__device__ void sum_partition(float *x, int i, int n, float *result) {
    int stride = __popc(n) == 1 ? n >> 1 : 1 << 31 - __clz(n);

    if(i + stride < n) {
        x[i] += x[i + stride];
    }
      
    __syncthreads();

    stride >>= 1;
    // max_stride necessary to keep all threads from all partitions in sync.
    for(int max_stride = Threads_Per_Block >> 4; max_stride > 0; stride >>= 1, max_stride >>= 1) {
        if(i < stride) {
            x[i] += x[i + stride];
        }
        __syncthreads();
    }

    if(i == 0) {
        *result += x[0];
    }

    __syncthreads();
}

static __device__ float logistic(float x, float slope) {
    return (1.0 / (1.0 + exp(-1 * x * slope)));
}

__global__ void update(FiringRateModel_Cuda::GpuState state) {
    int tid = threadIdx.x;

    extern __shared__ char __shared_buf[];

    float *neuronactivation = (float *)__shared_buf;
    float *newneuronactivation = neuronactivation + state.neurons_count;
    float *partial_activation = newneuronactivation + state.neurons_count;

    FiringRateModel_Cuda::Neuron neuron;
    if(tid < state.neurons_count) {
        neuron = state.buffers.neurons[tid];
        neuronactivation[tid] = state.buffers.neuronactivation[tid];
        newneuronactivation[tid] = neuron.bias;
    }
    __syncthreads();

    FiringRateModel_Cuda::Synapse synapses[MAX_SYNAPSES_PER_THREAD];

    for(int i = tid, it = 0; i < state.synapses_count; i += Threads_Per_Block, it++) {
        synapses[it] = state.buffers.synapses[i];
        FiringRateModel_Cuda::NeuronActivationPartition p = state.buffers.partitions[synapses[it].partition];

        partial_activation[tid] = synapses[it].efficacy * neuronactivation[synapses[it].fromneuron];
        __syncthreads();

        sum_partition(partial_activation + p.offset,
                      tid - p.offset,
                      p.len,
                      newneuronactivation + p.toneuron);
        __syncthreads();
    }

    if(tid < state.neurons_count) {
        newneuronactivation[tid] =
            (1.0f - neuron.tau) * neuronactivation[tid]
            + neuron.tau * logistic( newneuronactivation[tid], state.logistic_slope );
    }
    __syncthreads();

    for(int i = tid; i < state.neurons_count; i += Threads_Per_Block) {
        state.buffers.newneuronactivation[i] = newneuronactivation[i];
    }
}

void FiringRateModel_Cuda::update(float *neuronactivation, float *newneuronactivation) {
    xcuda( cudaMemcpy(gpu.buffers.neuronactivation, neuronactivation, sizeof(float)*gpu.neurons_count, cudaMemcpyHostToDevice) );

    size_t sizeof_shared = (2 * gpu.neurons_count + Threads_Per_Block) * sizeof(float);

    ::update<<<1, Threads_Per_Block, sizeof_shared>>>( gpu );

    float test[gpu.neurons_count * sizeof(float)];
    xcuda( cudaMemcpy(test, gpu.buffers.newneuronactivation, sizeof(float)*gpu.neurons_count, cudaMemcpyDeviceToHost) );
    for(int i = gpu.input_neurons_count; i < gpu.neurons_count; i++) {
        if(fabs((test[i] - newneuronactivation[i]) / newneuronactivation[i]) > 0.00001) {
            std::cerr << "bad neuron " << i << ": expected=" << newneuronactivation[i] << ", actual=" << test[i] << std::endl;
        }
    }
    exit(0);
}
