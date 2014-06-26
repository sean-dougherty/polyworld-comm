#pragma once

#include <assert.h>

#include <iostream>
#include <memory>

#include "Genome.h"

//===========================================================================
// FitStruct
//===========================================================================
struct FitStruct
{
	unsigned long	agentID;
	float	fitness;
	float   complexity;
    std::shared_ptr<genome::Genome> genes;
};
typedef struct FitStruct FitStruct;

//===========================================================================
// FittestList
//===========================================================================
class FittestList
{
 public:
	FittestList( int capacity, bool storeGenome );
	virtual ~FittestList();

	int update( class agent *candidate, float fitness );
	int update( FitStruct *fs );

	bool isFull();
	void clear();
	int size();
    int capacity();
    void dropLast();

	// rank is 0-based
	FitStruct *get( int rank );

	void dump( std::ostream &out );

 private:
	int _capacity;
	bool _storeGenome;
	int _size;

 public: // tmp access
	FitStruct **_elements;
};

inline bool FittestList::isFull() { return _size == _capacity; }
inline int FittestList::size() { return _size; }
inline int FittestList::capacity() { return _capacity; }
inline FitStruct *FittestList::get( int rank ) { assert(rank < _size); return _elements[rank]; }
