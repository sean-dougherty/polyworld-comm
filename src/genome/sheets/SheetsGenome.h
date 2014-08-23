#pragma once

#include "Genome.h"

namespace genome
{
	class SheetsGenome : public Genome
	{
	public:
		SheetsGenome( class SheetsGenomeSchema *schema,
					  GenomeLayout *layout );
		virtual ~SheetsGenome();

		Brain *createBrain( NervousSystem *cns );

		virtual void crossover( Genome *g1,
								Genome *g2,
								bool mutate,
                                float rate_multiplier = 1.0f) override;

	private:
		SheetsGenomeSchema *_schema;
	};
}
