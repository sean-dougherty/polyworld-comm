#pragma once

#include "BaseNeuronModel.h"
#include "FiringRateModel_Common.h"
#include "FiringRateModel_Cuda.h"

// forward decls
class NervousSystem;

class FiringRateModel : public BaseNeuronModel<FiringRateModel__Neuron, FiringRateModel__NeuronAttrs, FiringRateModel__Synapse>
{
	typedef FiringRateModel__Neuron Neuron;
	typedef FiringRateModel__NeuronAttrs NeuronAttrs;
	typedef FiringRateModel__Synapse Synapse;

 public:
	FiringRateModel( NervousSystem *cns );
	virtual ~FiringRateModel();

	virtual void init_derived( float initial_activation );

	virtual void set_neuron( int index,
							 void *attributes,
							 int startsynapses,
							 int endsynapses );

    virtual void complete();

	virtual void update( bool bprint );

private:
    FiringRateModel_Cuda cuda;
};
