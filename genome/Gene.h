#pragma once

#include <list>
#include <map>
#include <string>
#include <vector>

#include "dynamic.h"
#include "NeurGroupType.h"
#include "Scalar.h"

namespace genome
{

	// forward decls
	class Genome;
	class GenomeLayout;
	class GenomeSchema;
	class SynapseType;

	// ================================================================================
	// ===
	// === CLASS Gene
	// ===
	// ================================================================================
	class Gene
	{
	public:
		enum Type
		{
			SCALAR,
			NEURGROUP,
			NEURGROUP_ATTR,
			SYNAPSE_ATTR
		};

		virtual ~Gene() {}

		void seed( Genome *genome,
				   unsigned char rawval );

		int getMutableSize();

		virtual void printIndexes( FILE *file, GenomeLayout *layout );
		virtual void printTitles( FILE *file );
		virtual void printRanges( FILE *file );

		Type type;
		std::string name;
		bool ismutable;

	protected:
		Gene() {}
		void init( Type type,
				   bool ismutable,
				   const char *name );

		virtual int getMutableSizeImpl();

		GenomeSchema *schema;

	protected:
		friend class GenomeSchema;
		friend class GenomeLayout;

		int offset;

	public:
#define CAST_TO(TYPE) \
		class TYPE##Gene *to_##TYPE();
	CAST_TO(NonVector);
	CAST_TO(ImmutableScalar);
	CAST_TO(MutableScalar);
	CAST_TO(ImmutableInterpolated);
	CAST_TO(NeurGroup);
	CAST_TO(NeurGroupAttr);
	CAST_TO(SynapseAttr);
	CAST_TO(__Interpolated);
#undef CAST_TO
	};


	// ================================================================================
	// ===
	// === CLASS __ConstantGene
	// ===
	// === Common base class for immutable gene types
	// ===
	// ================================================================================
	class __ConstantGene : virtual Gene
	{
	public:
		__ConstantGene( Type type,
						const char *name,
						const Scalar &value );
		virtual ~__ConstantGene() {}

		const Scalar &get();

	private:
		const Scalar value;
	};


	// ================================================================================
	// ===
	// === CLASS __InterpolatedGene
	// ===
	// === Common base class for genes whose values are derived from a range.
	// === 
	// ================================================================================
	class __InterpolatedGene : virtual Gene
	{
		PROPLIB_DYNAMIC_PROPERTIES

	public:
		enum Rounding
		{
			ROUND_NONE,
			ROUND_INT_FLOOR,
			ROUND_INT_NEAREST,
			ROUND_INT_BIN
		};

		__InterpolatedGene( Type type,
							bool ismutable,
							const char *name,
							Gene *gmin,
							Gene *gmax,
							Rounding rounding );
		virtual ~__InterpolatedGene() {}

		const Scalar &getMin();
		const Scalar &getMax();

		Scalar interpolate( unsigned char raw );
		Scalar interpolate( double ratio );

		void printRanges( FILE *file );

	private:
		Scalar smin;
		Scalar smax;
		Rounding rounding;
	};


	// ================================================================================
	// ===
	// === CLASS NonVectorGene
	// ===
	// === Genome interface; common for gene types that do not contain
	// === multiple elements.
	// ===
	// ================================================================================
	class NonVectorGene : virtual public Gene
	{
	public:
		virtual ~NonVectorGene() {}

		virtual Scalar get( Genome *genome ) = 0;

	protected:
		virtual int getMutableSizeImpl();
	};


	// ================================================================================
	// ===
	// === CLASS ImmutableScalarGene
	// ===
	// === For public use; holds constant values.
	// ===
	// ================================================================================
	class ImmutableScalarGene : public NonVectorGene, private __ConstantGene
	{
	public:
		ImmutableScalarGene( const char *name,
							 const Scalar &value );
		virtual ~ImmutableScalarGene() {}

		virtual Scalar get( Genome *genome );
	};


	// ================================================================================
	// ===
	// === CLASS MutableScalarGene
	// ===
	// === For public use; holds single mutable value interpolated over a range.
	// ===
	// ================================================================================
	class MutableScalarGene : public NonVectorGene, public __InterpolatedGene
	{
	public:
		MutableScalarGene( const char *name,
						   Gene *gmin,
						   Gene *gmax,
						   __InterpolatedGene::Rounding rounding );
		virtual ~MutableScalarGene() {}

		virtual Scalar get( Genome *genome );

		const Scalar &getMin();
		const Scalar &getMax();

		virtual void printRanges( FILE *file );
	};


	// ================================================================================
	// ===
	// === CLASS ImmutableInterpolatedGene
	// ===
	// === For public use; the "raw" value used for interpolation between min and
	// === max is provided externally. That is, the "raw" value does not come from
	// === the mutable data of the Genome.
	// ===
	// ================================================================================
	class ImmutableInterpolatedGene : virtual public Gene, public __InterpolatedGene
	{
	public:
		ImmutableInterpolatedGene( const char *name,
								   Gene *gmin,
								   Gene *gmax,
								   __InterpolatedGene::Rounding rounding );
		virtual ~ImmutableInterpolatedGene() {}

		Scalar interpolate( double ratio );
	};


	// ================================================================================
	// ===
	// === CLASS NeurGroupGene
	// ===
	// === Common base class for neural group genes.
	// ===
	// ================================================================================
	class NeurGroupGene : public NonVectorGene
	{
	public:
		NeurGroupGene( NeurGroupType group_type );				   
		virtual ~NeurGroupGene() {}

		virtual Scalar get( Genome *genome ) = 0;

		bool isMember( NeurGroupType group_type );
		NeurGroupType getGroupType();

		virtual int getMaxGroupCount() = 0;
		virtual int getMaxNeuronCount() = 0;

		virtual std::string getTitle( int group ) = 0;

	protected:
		friend class GenomeSchema;

		int first_group;

	private:
		NeurGroupType group_type;
	};


	// ================================================================================
	// ===
	// === CLASS MutableNeurGroupGene
	// ===
	// === For public use; represents Internal groups as well as Input groups with
	// === variable neuron counts
	// ===
	// ================================================================================
	class MutableNeurGroupGene : public NeurGroupGene, public __InterpolatedGene
	{
	public:
		MutableNeurGroupGene( const char *name,
							  NeurGroupType group_type,
							  Gene *gmin,
							  Gene *gmax );
		virtual ~MutableNeurGroupGene() {}

		virtual Scalar get( Genome *genome );

		virtual int getMaxGroupCount();
		virtual int getMaxNeuronCount();

		virtual std::string getTitle( int group );
	};


	// ================================================================================
	// ===
	// === CLASS ImmutableNeurGroupGene
	// ===
	// === For public use; represents Input groups with fixed neuron counts as well as
	// === Output groups.
	// ===
	// ================================================================================
	class ImmutableNeurGroupGene : public NeurGroupGene, private __ConstantGene
	{
	public:
		ImmutableNeurGroupGene( const char *name,
								NeurGroupType group_type );
		virtual ~ImmutableNeurGroupGene() {}

		virtual Scalar get( Genome *genome );

		virtual int getMaxGroupCount();
		virtual int getMaxNeuronCount();

		virtual std::string getTitle( int group );
	};


	// ================================================================================
	// ===
	// === CLASS NeurGroupAttrGene
	// ===
	// === For public use; provides a per-group attribute vector
	// ===
	// ================================================================================
	class NeurGroupAttrGene : virtual public Gene, public __InterpolatedGene
	{
	public:
		NeurGroupAttrGene( const char *name,
						   NeurGroupType group_type,
						   Gene *gmin,
						   Gene *gmax );
		virtual ~NeurGroupAttrGene() {}

		virtual Scalar get( Genome *genome,
							int group );

		const Scalar &getMin();
		const Scalar &getMax();

		void seed( Genome *genome,
				   NeurGroupGene *group,
				   unsigned char rawval );

		virtual void printIndexes( FILE *file, GenomeLayout *layout );
		virtual void printTitles( FILE *file );

		const NeurGroupType group_type;

	protected:
		virtual int getMutableSizeImpl();

		friend class GenomeLayout;

		int getOffset( int group );
	};

	// ================================================================================
	// ===
	// === CLASS SynapseAttrGene
	// ===
	// === For public use; provides a per-synapse attribute matrix
	// ===
	// ================================================================================
	class SynapseAttrGene : virtual public Gene, public __InterpolatedGene
	{
	public:
		SynapseAttrGene( const char *name,
						 bool negateInhibitory,
						 bool lessThanZero,
						 Gene *gmin,
						 Gene *gmax );
		virtual ~SynapseAttrGene() {}

		virtual Scalar get( Genome *genome,
							SynapseType *synapseType,
							int group_from,
							int group_to );

		void seed( Genome *genome,
				   SynapseType *synapseType,
				   NeurGroupGene *from,
				   NeurGroupGene *to,
				   unsigned char rawval );

		virtual void printIndexes( FILE *file, GenomeLayout *layout );
		virtual void printTitles( FILE *file );

	protected:
		virtual int getMutableSizeImpl();

		friend class GenomeLayout;

		int getOffset( SynapseType *synapseType,
					   int from,
					   int to );

	private:
		bool negateInhibitory;
		bool lessThanZero;
	};


	// ================================================================================
	// ===
	// === Collections
	// ===
	// ================================================================================
	typedef std::vector<Gene *> GeneVector;
	typedef std::map<std::string, Gene *> GeneMap;
	typedef std::list<Gene *> GeneList;
	typedef std::map<Gene::Type, GeneVector> GeneTypeMap;

} // namespace genome
