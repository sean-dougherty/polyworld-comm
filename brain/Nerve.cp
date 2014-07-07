#include "Nerve.h"

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
