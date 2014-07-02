#include "FiringRateModel_Cuda.h"

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>

#define Threads_Per_Block 512

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

    FREE(bias);

    FREE(partitions);
    FREE(endpoints);
    FREE(efficacy);
    FREE(lrate);
    FREE(partition_index);
    FREE(neuronactivation);
    FREE(newneuronactivation);
}

void FiringRateModel_Cuda::init(FiringRateModel__Neuron *neurons,
                                short neurons_count, short input_neurons_count,
                                FiringRateModel__Synapse *synapses,
                                long synapses_count) {
    gpu.neurons_count = neurons_count;
    gpu.input_neurons_count = input_neurons_count;
    gpu.synapses_count = synapses_count;

    float bias[neurons_count];

    for(short i = 0; i < neurons_count; i++) {
        bias[i] = neurons[i].bias;
    }

    NeuronSynapsesPartition partitions[synapses_count];
    NeuronSynapsesPartition *currpartition = NULL;
    SynapseEndpoints endpoints[synapses_count];
    float efficacy[synapses_count];
    float lrate[synapses_count];
    long partition_index[synapses_count];

    for(long i = 0; i < synapses_count; i++) {
        FiringRateModel__Synapse &synapse = synapses[i];
        if( (i % Threads_Per_Block == 0) || (synapse.toneuron != currpartition->toneuron) ) {
            if(currpartition)
                currpartition++;
            else
                currpartition = partitions;
            currpartition->toneuron = synapse.toneuron;
            currpartition->offset = i % Threads_Per_Block;
            currpartition->len = 0;
        }
        currpartition->len++;

        // Store synapse data in different regions in order to reduce cache pressure in GPU.
        // This way, we only load data at the point we need it.
        endpoints[i].toneuron = synapse.toneuron;
        endpoints[i].fromneuron = synapse.fromneuron;
        efficacy[i] = synapse.efficacy;
        lrate[i] = synapse.lrate;
        partition_index[i] = currpartition - partitions;
    }

    size_t npartitions = currpartition - partitions + 1;

/*
    for(long i = 0; i < synapses_count; i++) {
        if( (i % Threads_Per_Block) == 0 ) {
            printf("********\n");
        }
        NeuronSynapsesPartition &p = partitions[partition_index[i]];
        assert(p.toneuron == synapses[i].toneuron);
        printf("%5ld %3d] %3d %3d %3d\n", i, synapses[i].toneuron, p.toneuron, p.offset, p.len);
    }
*/

    for(size_t i = 0; i < npartitions; i++) {
        NeuronSynapsesPartition &p = partitions[i];
        printf("%4lu] to=%3d off=%4d len=%3d\n", i, p.toneuron, p.offset, p.len);
    }


    size_t sizeof_partitions = npartitions * sizeof(NeuronSynapsesPartition);
    xcuda( cudaMalloc((void**)&gpu.buffers.partitions, sizeof_partitions) );
    xcuda( cudaMemcpy(gpu.buffers.partitions, partitions, sizeof_partitions, cudaMemcpyHostToDevice) );

#define alloc(var) \
    xcuda( cudaMalloc((void**)&gpu.buffers.var, sizeof(var)) );                  \
    xcuda( cudaMemcpy(gpu.buffers.var, var, sizeof(var), cudaMemcpyHostToDevice) ); \
    assert( gpu.buffers.var != NULL );

    alloc(bias);

    alloc(endpoints);
    alloc(efficacy);
    alloc(lrate);
    alloc(partition_index);

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

__global__ void update(FiringRateModel_Cuda::GpuState state) {
    int tid = threadIdx.x;

    extern __shared__ char __shared_buf[];

    float *neuronactivation = (float *)__shared_buf;
    float *newneuronactivation = neuronactivation + state.neurons_count;
    float *partial_activation = newneuronactivation + state.neurons_count;

    for(int i = tid; i < state.neurons_count; i += Threads_Per_Block) {
        neuronactivation[i] = state.buffers.neuronactivation[i];
        newneuronactivation[i] = state.buffers.bias[i];
    }
    __syncthreads();

    for(int i = tid; i < state.synapses_count; i += Threads_Per_Block) {
        FiringRateModel_Cuda::SynapseEndpoints endpoint = state.buffers.endpoints[i];
        float efficacy = state.buffers.efficacy[i];
        long partition_index = state.buffers.partition_index[i];
        FiringRateModel_Cuda::NeuronSynapsesPartition p = state.buffers.partitions[partition_index];

        partial_activation[tid] = efficacy * neuronactivation[endpoint.fromneuron];
        __syncthreads();


        sum_partition(partial_activation + p.offset,
                      tid - p.offset,
                      p.len,
                      newneuronactivation + p.toneuron);
        __syncthreads();
    }

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
