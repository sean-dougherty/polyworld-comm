#pragma once

#include <assert.h>

#include <iostream>

#include "GenomeLayout.h"
#include "GenomeSchema.h"
#include "graybin.h"

// forward decl
class AbstractFile;
class Brain;
class NervousSystem;
class RandomNumberGenerator;

namespace genome
{
	// forward decl
	class GenomeLayout;

	// ================================================================================
	// ===
	// === CLASS Genome
	// ===
	// ================================================================================
	class Genome
	{
	public:
		Genome( GenomeSchema *schema,
				GenomeLayout *layout );
		virtual ~Genome();

        RandomNumberGenerator *getRNG();

		virtual Brain *createBrain( NervousSystem *cns ) = 0;

		Gene *MISC_BIAS;
		Gene *MISC_INVIS_SLOPE;

		Gene *gene( const char *name );

		Scalar get( const char *name );
		Scalar get( Gene *gene );

		unsigned int get_raw_uint( long byte );
		void updateSum( unsigned long *sum, unsigned long *sum2 );

		void seed( Gene *gene,
				   float rawval_ratio );

		void randomize( float bitonprob );
		void randomize();

		void mutate();
		virtual void crossover( Genome *g1,
								Genome *g2,
								bool mutate );
		void copyFrom( Genome *g );
		float separation( Genome *g );
		float mateProbability( Genome *g );

		void dump( AbstractFile *out );
		void load( AbstractFile *in );

		void print();
		void print( long lobit, long hibit );

        //protected: todo, restore protected. made public for development of mpi stuff.
		friend class Gene;
		friend class MutableScalarGene;
		friend class MutableNeurGroupGene;
		friend class NeurGroupAttrGene;
		friend class SynapseAttrGene;

		virtual void getCrossoverPoints( long *crossoverPoints, long numCrossPoints );

		unsigned char get_raw( int offset );
		void set_raw( int offset,
					  int n,
					  unsigned char rawval );

		int nbytes;
		unsigned char *mutable_data;

	private:
		void alloc();

        RandomNumberGenerator *rng;
		GenomeSchema *schema;
		GenomeLayout *layout;
		bool gray;
	};


//===========================================================================
// inlines
//===========================================================================
inline Scalar Genome::get( Gene *gene )
{
	return GeneType::to_NonVector( gene )->get( this );
}

inline Gene *Genome::gene( const char *name )
{
	return schema->get( name );
}

inline Scalar Genome::get( const char *name )
{
	return get( gene(name) );
}

inline unsigned int Genome::get_raw_uint( long byte )
{
	return (unsigned int)get_raw( byte );
}

inline unsigned char Genome::get_raw( int offset )
{
	assert( offset >= 0 && offset < nbytes );

	int layoutOffset = layout->getMutableDataOffset( offset );
	unsigned char val = mutable_data[layoutOffset];
	if( gray )
	{
		val = binofgray[val];
	}

	return val;
}

inline void Genome::set_raw( int offset,
							 int n,
							 unsigned char val )
{
	assert( (offset >= 0) && (offset + n <= nbytes) );

	if( gray )
	{
		val = grayofbin[val];
	}

	for( int i = 0; i < n; i++ )
	{
		int layoutOffset = layout->getMutableDataOffset( offset + i );

		mutable_data[layoutOffset] = val;
	}
}

} // namespace genome
