#include "FiringRateModel.h"

#include "agent.h"
#include "debug.h"
#include "Genome.h"
#include "GenomeSchema.h"
#include "Logs.h"
#include "misc.h"
#include "simtypes.h"
#include "timer.h"

#include "Brain.h" // temporary

#include <array>

using namespace std;
using namespace sim;

using namespace genome;

double brain_time = 0.0;

typedef FiringRateModel_Cuda::AgentState AgentState;
typedef FiringRateModel_Cuda::GpuState GpuState;

static bool changed = false;
static AgentState *agents = nullptr;
static long nagents = 0;

void FiringRateModel::update()
{
    double start = seconds();

    if(changed) {
        if(agents)
            delete [] agents;

        nagents = objectxsortedlist::gXSortedObjects.getCount(AGENTTYPE);
        agents = new AgentState[nagents];
        {
            objectxsortedlist::gXSortedObjects.reset();
            long i = 0;
            agent *a;
            while (objectxsortedlist::gXSortedObjects.nextObj(AGENTTYPE, (gobject**)&a)) {
                FiringRateModel_Cuda::AgentState &as = agents[i++];
                FiringRateModel *model = dynamic_cast<FiringRateModel *>(a->GetBrain()->_neuralnet);
                as.a = a;
                as.model = &model->cuda;
                as.neuronactivation = model->neuronactivation;
                as.newneuronactivation = model->newneuronactivation;
            }
        }
    }

    for(long i = 0; i < nagents; i++) {
        agents[i].a->GetNervousSystem()->update(false);
    }

    FiringRateModel_Cuda::update(agents, nagents);

    for(long i = 0; i < nagents; i++) {
        logs->postEvent( BrainUpdatedEvent(agents[i].a) );
    }

    brain_time += seconds() - start;
}

FiringRateModel::FiringRateModel( NervousSystem *cns )
: BaseNeuronModel<Neuron, NeuronAttrs, Synapse>( cns )
{
    assert(Brain::config.neuronModel == Brain::Configuration::TAU);
    changed = true;
}

FiringRateModel::~FiringRateModel()
{
    changed = true;
}

void FiringRateModel::init_derived( float initial_activation )
{
	for( int i = 0; i < dims->numNeurons; i++ )
		neuronactivation[i] = initial_activation;
}

void FiringRateModel::set_neuron( int index,
								  void *attributes,
								  int startsynapses,
								  int endsynapses )
{
	BaseNeuronModel<Neuron, NeuronAttrs, Synapse>::set_neuron( index,
															   attributes,
															   startsynapses,
															   endsynapses );
	NeuronAttrs *attrs = (NeuronAttrs *)attributes;
	Neuron &n = neuron[index];

	assert( !isnan(attrs->tau) );
	n.tau = attrs->tau;
}

void FiringRateModel::complete()
{
    cuda.init(neuron, dims->numNeurons, dims->numInputNeurons, dims->numOutputNeurons,
              neuronactivation,
              synapse, dims->numSynapses,
              Brain::config.logisticSlope,
              Brain::config.decayRate,
              Brain::config.maxWeight);
}

void FiringRateModel::update( bool bprint )
{
    assert(false); // NEED TO ADAPT TO NEW SINGLE ACTIVATION BUFFER.
    debugcheck( "(firing-rate brain) on entry" );

    if ((neuron == NULL) || (synapse == NULL) || (neuronactivation == NULL))
        return;

	long numneurons = dims->numNeurons;
	long numsynapses = dims->numSynapses;
	float logisticSlope = Brain::config.logisticSlope;

    for( short i = dims->getFirstOutputNeuron(); i < numneurons; i++ )
    {
		newneuronactivation[i] = neuron[i].bias;
    }

    for( long k = 0; k < numsynapses; k++ ) {
        FiringRateModel__Synapse &syn = synapse[k];

        float fromactivation = neuronactivation[syn.fromneuron];
        newneuronactivation[syn.toneuron] += syn.efficacy * fromactivation;
    }

    for( short i = dims->getFirstOutputNeuron(); i < numneurons; i++ )
    {
		float newactivation = newneuronactivation[i];
        float tau = neuron[i].tau;
        newactivation = (1.0 - tau) * neuronactivation[i]  +  tau * logistic( newactivation, logisticSlope );
        newneuronactivation[i] = newactivation;
    }

    debugcheck( "after updating neurons" );

    float learningrate;
    for (long k = 0; k < numsynapses; k++)
    {
		FiringRateModel__Synapse &syn = synapse[k];

		learningrate = syn.lrate;
        
		float efficacy = syn.efficacy + learningrate
			* (newneuronactivation[syn.toneuron]-0.5f)
			* (   neuronactivation[syn.fromneuron]-0.5f);

        if (fabs(efficacy) > (0.5f * Brain::config.maxWeight))
        {
            efficacy *= 1.0f - (1.0f - Brain::config.decayRate) *
                (fabs(efficacy) - 0.5f * Brain::config.maxWeight) / (0.5f * Brain::config.maxWeight);
            if (efficacy > Brain::config.maxWeight)
                efficacy = Brain::config.maxWeight;
            else if (efficacy < -Brain::config.maxWeight)
                efficacy = -Brain::config.maxWeight;
        }
        else
        {
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))
            // not strictly correct for this to be in an else clause,
            // but if lrate is reasonable, efficacy should never change
            // sign with a new magnitude greater than 0.5 * Brain::config.maxWeight
            if (learningrate >= 0.0f)  // excitatory
                efficacy = MAX(0.0f, efficacy);
            if (learningrate < 0.0f)  // inhibitory
                efficacy = MIN(-1.e-10f, efficacy);
        }

		syn.efficacy = efficacy;
    }

    debugcheck( "after updating synapses" );

    float* saveneuronactivation = neuronactivation;
    neuronactivation = newneuronactivation;
    newneuronactivation = saveneuronactivation;
}
