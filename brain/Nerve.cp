#include "Nerve.h"

#include <assert.h>
#include <string.h>

using namespace std;

Nerve::Nerve( Type _type,
			  const std::string &_name,
			  int _igroup )
: type(_type)
, name(_name)
, numneurons(0)
, index(-1)
{
    activations = nullptr;
}

float Nerve::get( int ineuron )
{
	if( numneurons == 0 )
		return 0.0;

	assert( (ineuron >= 0) && (ineuron < numneurons) && (index > -1) );
	
	return activations[index + ineuron];
}

void Nerve::set( float activation )
{
	if( numneurons == 0 )
		return;

	assert( numneurons == 1 );
	
	activations[index] = activation;
}

void Nerve::set( int ineuron,
				 float activation )
{
	assert( (ineuron >= 0) && (ineuron < numneurons) && (index > -1) );

	activations[index + ineuron] = activation;
}

int Nerve::getIndex()
{
	return index;
}

int Nerve::getNeuronCount()
{
	return numneurons;
}

void Nerve::config( int _numneurons,
					int _index )
{
	numneurons = _numneurons;
	index = _index;
}

void Nerve::config( float *activations_ )
{
	activations = activations_;
}
