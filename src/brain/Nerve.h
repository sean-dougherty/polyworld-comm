#pragma once

#include <assert.h>
#include <string>

class Nerve
{
 public:
	enum Type
	{
		INPUT = 0,
		OUTPUT,
		__NTYPES
	};

	const Type type;
	const std::string name;

 private:
	friend class NervousSystem;

	Nerve( Type type,
		   const std::string &name,
		   int igroup );

 public:
	float get( int ineuron = 0 );
	void set( float activation );
	void set( int ineuron,
			  float activation );
	int getIndex();
	int getNeuronCount();

 public:
	void config( int numneurons,
				 int index );
	void config( float *activations );
	
 private:
	int numneurons;
	float *activations;
	int index;
};

inline void Nerve::set( int ineuron,
                        float activation )
{
	assert( (ineuron >= 0) && (ineuron < numneurons) && (index > -1) );

	activations[index + ineuron] = activation;
}

inline float Nerve::get( int ineuron )
{
	if( numneurons == 0 )
		return 0.0;

	assert( (ineuron >= 0) && (ineuron < numneurons) && (index > -1) );
	
	return activations[index + ineuron];
}

inline void Nerve::set( float activation )
{
	if( numneurons == 0 )
		return;

	assert( numneurons == 1 );
	
	activations[index] = activation;
}
