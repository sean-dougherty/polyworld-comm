#include "FittestList.h"

#include "agent.h"
#include "GenomeUtil.h"

using namespace genome;
using namespace std;


//===========================================================================
// FittestList
//===========================================================================

//---------------------------------------------------------------------------
// FittestList::FittestList
//---------------------------------------------------------------------------
FittestList::FittestList( int capacity, bool storeGenome )
: _capacity( capacity )
, _storeGenome( storeGenome )
, _size( 0 )
{
	_elements = new FitStruct*[ _capacity ];
	for( int i = 0; i < _capacity; i++ )
	{
		FitStruct *element = new FitStruct;
		_elements[i] = element;
	}

	clear();
}

//---------------------------------------------------------------------------
// FittestList::~FittestList
//---------------------------------------------------------------------------
FittestList::~FittestList()
{
	for( int i = 0; i < _capacity; i++ )
	{
		delete _elements[i];
	}
	delete [] _elements;
}

//---------------------------------------------------------------------------
// FittestList::update
//---------------------------------------------------------------------------
int FittestList::update( agent *candidate, float fitness )
{
    FitStruct fs = {(unsigned long)candidate->Number(), fitness, candidate->Complexity(), candidate->Genes()};
    return update( &fs );
}

//---------------------------------------------------------------------------
// FittestList::update
//---------------------------------------------------------------------------
int FittestList::update( FitStruct *fs )
{
	if( !isFull() || (_capacity > 0 && fs->fitness > _elements[_size - 1]->fitness) )
	{
		int rank = -1;
		for( int i = 0; i < _size; i++ )
		{
			if( fs->fitness >= _elements[i]->fitness )
			{
				rank = i;
				break;
			}
		}
		if( rank == -1 )
		{
			assert( !isFull() );
			rank = _size;
		}

		if( !isFull() )
		{
			_size++;
		}

		FitStruct *newElement = _elements[_size-1];
		for( int i = _size - 1; i > rank; i-- )
			_elements[i] = _elements[i-1];
		_elements[rank] = newElement;

        *newElement = *fs;

        return rank;
	}
    else
    {
        return -1;
    }
}

//---------------------------------------------------------------------------
// FittestList::dropLast
//---------------------------------------------------------------------------
void FittestList::dropLast()
{
    if( _size > 0 )
        _size--;
}

//---------------------------------------------------------------------------
// FittestList::clear
//---------------------------------------------------------------------------
void FittestList::clear()
{
	_size = 0;

	// todo: this shouldn't be necessary. retained to ensure backwards-compatible
	for( int i = 0; i < _capacity; i++ )
	{
		_elements[i]->fitness = 0.0;
		_elements[i]->agentID = 0;
		_elements[i]->complexity = 0.0;
	}
}

//---------------------------------------------------------------------------
// FittestList::dump
//---------------------------------------------------------------------------
void FittestList::dump( ostream &out )
{
	for( int i = 0; i < _capacity; i++ )
	{
		out << _elements[i]->agentID << endl;
		out << _elements[i]->fitness << endl;
		out << _elements[i]->complexity << endl;
		
		assert( false );
		/* PORT TO AbstractFile
		if( _storeGenome )
			_elements[i]->dump(out);
		*/
	}
}
