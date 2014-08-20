#pragma once

#include <list>
#include <map>
#include <string>
#include <vector>

#include "cppprops.h"
#include "NeurGroupType.h"
#include "Scalar.h"

namespace genome
{

	// forward decls
	class Genome;
	class GenomeLayout;
	class GenomeSchema;

	// ================================================================================
	// ===
	// === Collections
	// ===
	// ================================================================================
	typedef std::vector<class Gene *> GeneVector;
	typedef std::map<std::string, class Gene *> GeneMap;
	typedef std::list<class Gene *> GeneList;
	typedef std::vector<GeneVector> GeneTypeLookup;

	// ================================================================================
	// ===
	// === CLASS GeneType
	// ===
	// ================================================================================
	class GeneType
	{
    private:
        static size_t id_counter;

	public:
		static const GeneType *SCALAR;
		static const GeneType *CONTAINER;

        const size_t id;

        GeneType();

#define CAST_TO(TYPE)											\
		static class TYPE##Gene *to_##TYPE(class Gene *gene);
		CAST_TO(NonVector);
		CAST_TO(ImmutableScalar);
		CAST_TO(MutableScalar);
		CAST_TO(ImmutableInterpolated);
		CAST_TO(__Interpolated);
		CAST_TO(Container);
#undef CAST_TO
	};

	// ================================================================================
	// ===
	// === CLASS Gene
	// ===
	// ================================================================================
	class Gene
	{
	public:
		virtual ~Gene() {}

		void seed( Genome *genome,
				   unsigned char rawval );

		int getMutableSize();
		int getOffset();

		virtual void printIndexes( FILE *file, const std::string &prefix, GenomeLayout *layout );
		virtual void printTitles( FILE *file, const std::string &prefix );
		virtual void printRanges( FILE *file, const std::string &prefix );

		const GeneType *type;
		std::string name;
		bool ismutable;

	protected:
		Gene() {}
		void init( const GeneType *type,
				   bool ismutable,
				   const char *name );

		virtual int getMutableSizeImpl();

	protected:
		friend class GeneSchema;
		friend class GenomeLayout;

		int offset;
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
		__ConstantGene( const GeneType *type,
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
        // dynamic properties no longer supported due to cached interpolation
		//PROPLIB_CPP_PROPERTIES

	public:
		enum Rounding
		{
			ROUND_NONE,
			ROUND_INT_FLOOR,
			ROUND_INT_NEAREST,
			ROUND_INT_BIN
		};

		__InterpolatedGene( const GeneType *type,
							bool ismutable,
							const char *name,
							Scalar min_,
							Scalar max_,
							Rounding rounding );
		virtual ~__InterpolatedGene() {}

		const Scalar &getMin();
		const Scalar &getMax();

		void setInterpolationPower( double interpolationPower );

		void printRanges( FILE *file, const std::string &prefix );

		Scalar interpolate( unsigned char raw );
		Scalar interpolate( double ratio );

	private:
        void updateCache();
		Scalar __interpolate( unsigned char raw );
		Scalar __interpolate( double ratio );

		Scalar smin;
		Scalar smax;
		Rounding rounding;
		double interpolationPower;
        Scalar cache[256];
	};

    inline Scalar __InterpolatedGene::interpolate( unsigned char raw )
    {
        return cache[raw];
    }

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
						   Scalar min_,
						   Scalar max_,
						   __InterpolatedGene::Rounding rounding );
		virtual ~MutableScalarGene() {}

		virtual Scalar get( Genome *genome );

		const Scalar &getMin();
		const Scalar &getMax();

		virtual void printRanges( FILE *file, const std::string &prefix );
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
								   Scalar min_,
								   Scalar max_,
								   __InterpolatedGene::Rounding rounding );
		virtual ~ImmutableInterpolatedGene() {}

		Scalar interpolate( double ratio );
	};


	// ================================================================================
	// ===
	// === CLASS ContainerGene
	// ===
	// ================================================================================
	class ContainerGene : public Gene
	{
	public:
		ContainerGene( const char *name );
		virtual ~ContainerGene();

		void add( Gene *gene );
		Gene *gene( const std::string &name );
		Scalar getConst( const std::string &name );
		const GeneVector &getAll();

		void complete();

		virtual void printIndexes( FILE *file, const std::string &prefix, GenomeLayout *layout );
		virtual void printTitles( FILE *file, const std::string &prefix );
		virtual void printRanges( FILE *file, const std::string &prefix );

	protected:
		virtual int getMutableSizeImpl();

	private:
		class GeneSchema *_containerSchema;
	};

	//===========================================================================
	// inlines
	//===========================================================================
#define CAST_TO(TYPE)											\
	inline TYPE##Gene *GeneType::to_##TYPE( Gene *gene_ )       \
	{															\
		TYPE##Gene *gene = dynamic_cast<TYPE##Gene *>( gene_ );	\
		assert( gene ); /* catch cast failure */				\
		return gene;											\
	}
	CAST_TO(NonVector);
	CAST_TO(ImmutableScalar);
	CAST_TO(MutableScalar);
	CAST_TO(ImmutableInterpolated);
	CAST_TO(__Interpolated);
	CAST_TO(Container);
#undef CAST_TO

} // namespace genome
