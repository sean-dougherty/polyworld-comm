/********************************************************************/
/* PolyWorld:  An Artificial Life Ecological Simulator              */
/* by Larry Yaeger                                                  */
/* Copyright Apple Computer 1990,1991,1992                          */
/********************************************************************/

// food.C - implementation of food classes

// Self
#include "food.h"

// System
#include <ostream>

// qt
#include <qapplication.h>

// Local
#include "agent.h"
#include "globals.h"
#include "graphics.h"
#include "Logs.h"
#include "simtypes.h"

using namespace std;

// Static class variables
unsigned long food::fFoodEver;

// External globals
float food::gFoodHeight;
Color food::gFoodColor;
float food::gMinFoodEnergy;
float food::gMaxFoodEnergy;
float food::gSize2Energy;
float food::gMaxFoodRadius;
float food::gCarryFood2Energy;
long food::gMaxLifeSpan;
bool food::gConstRadius;
food::FoodList food::gAllFood;

//===========================================================================
// food
//===========================================================================


//-------------------------------------------------------------------------------------------
// food::food
//-------------------------------------------------------------------------------------------
food::food( const FoodType *foodType, long step )
{
	initfood( foodType, step );
}


//-------------------------------------------------------------------------------------------
// food::food
//-------------------------------------------------------------------------------------------
food::food( const FoodType *foodType, long step, const Energy &e )
{
	initfood( foodType, step, e );
}


//-------------------------------------------------------------------------------------------
// food::food
//-------------------------------------------------------------------------------------------
food::food( const FoodType *foodType, long step, const Energy &e, float x, float z )
{
	initfood( foodType, step, e, x, z );
}


//-------------------------------------------------------------------------------------------
// food::~food
//-------------------------------------------------------------------------------------------
food::~food()
{
	assert( *fAllFoodIterator == this );
	gAllFood.erase( fAllFoodIterator );

    logs->postEvent( sim::FoodDeathEvent(this) );
}


//-------------------------------------------------------------------------------------------
// food::dump
//-------------------------------------------------------------------------------------------
void food::dump(ostream& out)
{
    panic(); //out << fEnergy nl;
    out << fPosition[0] sp fPosition[1] sp fPosition[2] nl;
}


//-------------------------------------------------------------------------------------------
// food::load
//-------------------------------------------------------------------------------------------
void food::load(istream& in)
{
    panic(); //in >> fEnergy;
    in >> fPosition[0] >> fPosition[1] >> fPosition[2];

    initlen();
}


//-------------------------------------------------------------------------------------------
// food::eat
//-------------------------------------------------------------------------------------------
Energy food::eat(const Energy &e)
{
	Energy actual = e;
	actual.constrain( 0, fEnergy );

	fEnergy -= actual;

    if( !gConstRadius )
        initlen();
	
	return actual;
}


//-------------------------------------------------------------------------------------------
// food::isDepleted
//-------------------------------------------------------------------------------------------
bool food::isDepleted()
{
	return fEnergy.isDepleted( foodType->depletionThreshold );
}

//-------------------------------------------------------------------------------------------
// food::getBirth
//-------------------------------------------------------------------------------------------
long food::getBirth()
{
	return fCreationStep;
}

//-------------------------------------------------------------------------------------------
// food::getAge
//-------------------------------------------------------------------------------------------
long food::getAge(long step)
{
	return step - fCreationStep;
}

//-------------------------------------------------------------------------------------------
// food::addCollaborator
//-------------------------------------------------------------------------------------------
void food::addCollaborator(long step)
{
    if( step != collaborators.step )
    {
        collaborators.step = step;
        collaborators.n = 1;
    }
    else
    {
        collaborators.n++;
    }
}

//-------------------------------------------------------------------------------------------
// food::getCollaboratorCount
//-------------------------------------------------------------------------------------------
long food::getCollaboratorCount( long step )
{
    if( step != collaborators.step )
        return 0;
    else
        return collaborators.n;
}

//-------------------------------------------------------------------------------------------
// food::initfood
//-------------------------------------------------------------------------------------------
void food::initfood( const FoodType *foodType, long step )
{
	Energy e = randpw() * (gMaxFoodEnergy - gMinFoodEnergy) + gMinFoodEnergy;
	initfood( foodType, step, e );
}


//-------------------------------------------------------------------------------------------
// food::initfood
//-------------------------------------------------------------------------------------------
void food::initfood( const FoodType *foodType, long step, const Energy &e )
{
	fEnergy = e;
	float x = randpw() * globals::worldsize;
	float z = randpw() * globals::worldsize;
	initfood( foodType, step, e, x, z );
}


//-------------------------------------------------------------------------------------------
// food::initfood
//-------------------------------------------------------------------------------------------
void food::initfood( const FoodType *foodType, long step, const Energy &e, float x, float z )
{
	this->foodType = foodType;
	fEnergy = e;
	initlen();
	fPosition[0] = x;
	fPosition[1] = 0.5 * fLength[1];
	fPosition[2] = z;
	initrest();

	fCreationStep = step;
	gAllFood.push_back( this );
	fAllFoodIterator = --gAllFood.end();
	assert( *fAllFoodIterator == this );
}
 

//-------------------------------------------------------------------------------------------
// food::initlen
//-------------------------------------------------------------------------------------------       
void food::initlen()
{
    if( gConstRadius )
    {
        setlen( gMaxFoodRadius, gFoodHeight, gMaxFoodRadius );
    }
    else
    {
        float lxz = 0.75 * fEnergy.mean() / gSize2Energy;
        float ly = gFoodHeight;
        setlen( lxz, ly, lxz );
    }
}


//-------------------------------------------------------------------------------------------
// food::initrest
//-------------------------------------------------------------------------------------------           
void food::initrest()
{
	setType( FOODTYPE );
	setTypeNumber( ++food::fFoodEver );
	setcolor( foodType->color );
    collaborators.step = -1;
}


//-------------------------------------------------------------------------------------------
// food::setradius
//-------------------------------------------------------------------------------------------           
void food::setradius()
{
	if( !fRadiusFixed )  //  only set radius anew if not set manually
		fRadius = sqrt( fLength[0]*fLength[0] + fLength[2]*fLength[2] ) * fRadiusScale * fScale * 0.5;
	srPrint( "food::%s(): r=%g%s\n", __FUNCTION__, fRadius, fRadiusFixed ? "(fixed)" : "" );
}
