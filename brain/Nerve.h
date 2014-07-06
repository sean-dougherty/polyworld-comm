#pragma once

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
