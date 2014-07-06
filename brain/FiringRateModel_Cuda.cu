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
    FREE(efficacy);
}

void FiringRateModel_Cuda::init(FiringRateModel__Neuron *neurons,
                                short neurons_count, short input_neurons_count, short output_neurons_count,
                                float *neuronactivation,
                                FiringRateModel__Synapse *synapses,
                                long synapses_count,
                                float logistic_slope,
                                float decay_rate,
                                float max_weight) {

    assert(neurons_count < MAX_NEURONS);

    gpu.neurons_count = neurons_count;
    gpu.input_neurons_count = input_neurons_count;
    gpu.output_neurons_count = output_neurons_count;
    gpu.synapses_count = synapses_count;
    gpu.logistic_slope = logistic_slope;
    gpu.decay_rate = decay_rate;
    gpu.max_weight = max_weight;

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
    float efficacy[synapses_count];

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
        gpu_synapse.lrate = synapse.lrate;

        efficacy[i] = synapse.efficacy;
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

    gpu.buffers.input_activation = NULL;

    size_t sizeof_activation = sizeof(float) * neurons_count;
    xcuda( cudaMalloc((void **)&gpu.buffers.neuronactivation, sizeof_activation) );
    xcuda( cudaMemcpy(gpu.buffers.neuronactivation, neuronactivation, sizeof_activation, cudaMemcpyHostToDevice) );

    xcuda( cudaMalloc((void**)&gpu.buffers.efficacy, sizeof(efficacy)) );
    xcuda( cudaMemcpy(gpu.buffers.efficacy, efficacy, sizeof(efficacy), cudaMemcpyHostToDevice) );
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

__global__ void update(FiringRateModel_Cuda::GpuState *states) {
    int tid = threadIdx.x;

    FiringRateModel_Cuda::GpuState state = states[blockIdx.x];

    extern __shared__ char __shared_buf[];

    float *neuronactivation = (float *)__shared_buf;
    float *newneuronactivation = neuronactivation + state.neurons_count;
    float *partial_activation = newneuronactivation + state.neurons_count;

    FiringRateModel_Cuda::Neuron neuron;
    if(tid < state.neurons_count) {
        neuron = state.buffers.neurons[tid];        
    	if(tid < state.input_neurons_count) {
        	neuronactivation[tid] = state.buffers.input_activation[tid];
    	} else {
			neuronactivation[tid] = state.buffers.neuronactivation[tid];
		}
		newneuronactivation[tid] = neuron.bias;
    }
    __syncthreads();

    FiringRateModel_Cuda::Synapse synapses[MAX_SYNAPSES_PER_THREAD];
    float efficacies[MAX_SYNAPSES_PER_THREAD];
    const int nits = 1 + (state.synapses_count - 1) / Threads_Per_Block;

    for(int i = tid, it = 0; it < nits; i += Threads_Per_Block, it++) {
        if(i < state.synapses_count) {
            synapses[it] = state.buffers.synapses[i];
            efficacies[it] = state.buffers.efficacy[i];
            partial_activation[tid] = efficacies[it] * neuronactivation[synapses[it].fromneuron];
        }
        __syncthreads();

        float *partition_x;
        int partition_i;
        int partition_n;
        float *result;
        
        if(i < state.synapses_count) {
            FiringRateModel_Cuda::NeuronActivationPartition p = state.buffers.partitions[synapses[it].partition];

            partition_x = partial_activation + p.offset;
            partition_i = tid - p.offset;
            partition_n = p.len;
            result = newneuronactivation + p.toneuron;
        } else {
            partition_x = NULL;
            partition_i = 1;
            partition_n = 0;
            result = NULL;
        }

        sum_partition(partition_x,
                      partition_i,
                      partition_n,
                      result);

        __syncthreads();
    }

    if(tid < state.neurons_count) {
        newneuronactivation[tid] =
            (1.0f - neuron.tau) * neuronactivation[tid]
            + neuron.tau * logistic( newneuronactivation[tid], state.logistic_slope );
    }
    __syncthreads();

    for(int i = tid, it = 0; i < state.synapses_count; i += Threads_Per_Block, it++) {
        FiringRateModel_Cuda::Synapse synapse = synapses[it];
        short toneuron = state.buffers.partitions[synapse.partition].toneuron;
        float efficacy = state.buffers.efficacy[i];

        efficacy += synapse.lrate
            * (newneuronactivation[toneuron] - 0.5f)
            * (neuronactivation[synapse.fromneuron] - 0.5f);

        if (abs(efficacy) > (0.5f * state.max_weight)) {
            efficacy *= 1.0f - (1.0f - state.decay_rate) *
                (abs(efficacy) - 0.5f * state.max_weight) / (0.5f * state.max_weight);
            if (efficacy > state.max_weight)
                efficacy = state.max_weight;
            else if (efficacy < -state.max_weight)
                efficacy = -state.max_weight;
        } else {
            // not strictly correct for this to be in an else clause,
            // but if lrate is reasonable, efficacy should never change
            // sign with a new magnitude greater than 0.5 * Brain::config.maxWeight
            if (synapse.lrate >= 0.0f)  // excitatory
                efficacy = max(0.0f, efficacy);
            if (synapse.lrate < 0.0f)  // inhibitory
                efficacy = min(-1.e-10f, efficacy);
        }

        state.buffers.efficacy[i] = efficacy;
    }

    if(tid < state.neurons_count) {
        state.buffers.neuronactivation[tid] = newneuronactivation[tid];

        if( (tid >= state.input_neurons_count)
            && (tid < state.input_neurons_count + state.output_neurons_count) ) {
            state.buffers.output_activation[tid - state.input_neurons_count] = newneuronactivation[tid];
        }
    }
}

typedef FiringRateModel_Cuda::AgentState AgentState;
typedef FiringRateModel_Cuda::GpuState GpuState;

static GpuState *d_gpus;
static uint sizeof_shared = 0;
static float *d_all_input = NULL;
static uint sizeof_input = 0;
static float *d_all_output = NULL;
static uint sizeof_output = 0;

void FiringRateModel_Cuda::alloc_update_buffers(AgentState *agents,
                                                long nagents,
                                                uint *input_offset,
                                                uint ninput,
                                                uint *output_offset,
                                                uint noutput) {
    if(d_all_input) {
        xcuda( cudaFree(d_gpus) );
        xcuda( cudaFree(d_all_input) );
        xcuda( cudaFree(d_all_output) );
    }

    sizeof_input = ninput * sizeof(float);
    sizeof_output = noutput * sizeof(float);

    sizeof_shared = 0;
    for(long i = 0; i < nagents; i++) {
        AgentState &agent = agents[i];
        GpuState *gpu = &agent.model->gpu;
        uint sizeof_agent = uint((2 * gpu->neurons_count + Threads_Per_Block) * sizeof(float));
        sizeof_shared = max(sizeof_shared, sizeof_agent);
    }

    xcuda( cudaMalloc((void**)&d_gpus, sizeof(GpuState) * nagents) );
    xcuda( cudaMalloc((void**)&d_all_input, sizeof_input) );
    xcuda( cudaMalloc((void**)&d_all_output, sizeof_output) );

    for(long i = 0; i < nagents; i++) {
        AgentState &agent = agents[i];
        GpuState *gpu = &agent.model->gpu;
        gpu->buffers.input_activation = d_all_input + input_offset[i];
        gpu->buffers.output_activation = d_all_output + output_offset[i];
    }
}

void FiringRateModel_Cuda::update_all(AgentState *agents,
                                      long nagents,
                                      float *all_input,
                                      float *all_output) {

    xcuda( cudaMemcpy(d_all_input,
                      all_input,
                      sizeof_input,
                      cudaMemcpyHostToDevice) );

    GpuState gpus[nagents];
    for(long i = 0; i < nagents; i++) {
        AgentState &agent = agents[i];
        GpuState *gpu = &agent.model->gpu;
        gpus[i] = *gpu;
    }

    xcuda( cudaMemcpy(d_gpus, gpus, sizeof(gpus), cudaMemcpyHostToDevice) );

    ::update<<<nagents, Threads_Per_Block, sizeof_shared>>>(d_gpus);

    xcuda( cudaMemcpy(all_output,
                      d_all_output,
                      sizeof_output,
                      cudaMemcpyDeviceToHost) );
}
