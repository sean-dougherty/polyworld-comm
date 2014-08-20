#pragma once

// note: activation levels are not maintained in the neuronstruct
// so that after the new activation levels are computed, the old
// and new blocks of memory can simply be repointered rather than
// copied.
struct FiringRateModel__Neuron
{
	float bias;
	float tau;
	long  startsynapses;
	long  endsynapses;
};

struct FiringRateModel__NeuronAttrs
{
	float bias;
	float tau;
};

struct FiringRateModel__Synapse
{
	float efficacy;   // > 0 for excitatory, < 0 for inhibitory
	float lrate;
	short fromneuron;
	short toneuron;
};

