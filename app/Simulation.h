#pragma once

#ifdef linux
	#include <errno.h>
#endif

#include <string>

// qt
#include <QObject>

// Local
#include "Domain.h"
#include "EatStatistics.h"
#include "Energy.h"
#include "Events.h"
#include "LifeSpan.h"
#include "Scheduler.h"
#include "SeparationCache.h"
#include "simconst.h"
#include "simtypes.h"
#include "gmisc.h"
#include "gpolygon.h"
#include "graphics.h"
#include "gstage.h"

using namespace sim;

// Forward declarations
namespace condprop { class PropertyList; }
namespace proplib { class Document; }
class TBinChartWindow;


//===========================================================================
// TSimulation
//===========================================================================

class TSimulation : public QObject
{
	Q_OBJECT

public:
	TSimulation( std::string worldfilePath, std::string monitorfilePath );
	virtual ~TSimulation();

	void Step();
	void End( const std::string &reason );
	std::string EndAt( long timestep );

	short WhichDomain( float x, float z, short d );
	void SwitchDomain( short newDomain, short oldDomain, int objectType );
	
	TBinChartWindow* GetGeneSeparationWindow() const;

	class AgentPovRenderer *GetAgentPovRenderer();
	class MonitorManager *getMonitorManager();
	gstage &getStage();

	bool isLockstep() const;
	long GetMaxAgents() const;
	static long fMaxNumAgents;
	long GetInitNumAgents() const;
	long getNumAgents( short domain = -1 );
	long GetNumDomains() const;

	long getNumBorn( AgentBirthType type );
	class agent *getCurrentFittest( int rank );
	float getFitnessWeight( FitnessWeightType type );
	float getFitnessStat( FitnessStatType type );
	float getFoodEnergyStat( FoodEnergyStatType type, FoodEnergyStatScope scope );
	
	void getStatusText( StatusText& statusText,
						int statusFrequency );

	long getStep() const;
	long GetMaxSteps() const;
	
	void MaintainEnergyCosts();
	double EnergyScaleFactor( long minAgents, long maxAgents, long numAgents );

	bool fLockStepWithBirthsDeathsLog;	// Are we running in lockstep mode?
	FILE * fLockstepFile;				// Define a file pointer to our LOCKSTEP-BirthsDeaths.log
	int fLockstepTimestep;				// Timestep at which the next event in LOCKSTEP-BirthDeaths.log occurs
	int fLockstepNumDeathsAtTimestep;	// How many agents died at this LockstepTimestep?
	int fLockstepNumBirthsAtTimestep;	// how many agents were born at LockstepTimestep?
	void SetNextLockstepEvent();		// function to read the next event from LOCKSTEP-BirthsDeaths.log


	static long fStep;
	static double fFramesPerSecondOverall;
	static double fSecondsPerFrameOverall;
	static double fFramesPerSecondRecent;
	static double fSecondsPerFrameRecent;
	static double fFramesPerSecondInstantaneous;
	static double fSecondsPerFrameInstantaneous;
	static double fTimeStart;

	int fBrainFunctionRecentRecordFrequency;
	long fCurrentRecentEpoch;
	
	int fBestSoFarBrainAnatomyRecordFrequency;
	int fBestSoFarBrainFunctionRecordFrequency;
	int fBestRecentBrainAnatomyRecordFrequency;
	int fBestRecentBrainFunctionRecordFrequency;
	
	bool fRecordBarrierPosition;

	bool fBrainAnatomyRecordAll;
	bool fBrainFunctionRecordAll;
	bool fBrainAnatomyRecordSeeds;
	bool fBrainFunctionRecordSeeds;
	
	bool fApplyLowPopulationAdvantage;
	bool fEnergyBasedPopulationControl;
	bool fPopControlGlobal;
	bool fPopControlDomains;
	double fPopControlMinFixedRange;
	double fPopControlMaxFixedRange;
	double fPopControlMinScaleFactor;
	double fPopControlMaxScaleFactor;
	double fGlobalEnergyScaleFactor;
	
	bool fRecordComplexity;				// record the Olaf Functional Complexity (neural)
	class DataLibWriter *fComplexityLog;
	class DataLibWriter *fComplexitySeedLog;
// 	class DataLibWriter *fAvrComplexityLog;

	float EnergyFitnessParameter() const;
	float AgeFitnessParameter() const;
	float LifeFractionRecent();
	unsigned long LifeFractionSamples();  // the number of samples upon which LifeFractionRecent() is based

	char glFogFunction();
	float glExpFogDensity();
	int glLinearFogEnd();


	float GetAgentHealingRate();				// Virgil Healing

	int getRandomPatch(int domainNumber);
	Domain fDomains[MAXDOMAINS];
	
	Events* fEvents;

signals:
	void ended();
	
private:
	void InitAgents();
	void InitNeuralValues();

	void SeedGenome( long agentNumber,
					 genome::Genome *genes,
					 float probabilityOfMutatingSeeds,
					 long numSeeded );
	void SeedGenomeFromFile( long agentNumber,
							 genome::Genome *genes,
							 long numSeeded );
	void ReadSeedFilePaths();

	void SetSeedPosition( agent *a,
						  long numSeeded,
						  float x, float y, float z );
	void ReadSeedPositionsFromFile();

	void InitComplexityLog( long epoch );
	void UpdateComplexityLog( agent *a );
	void EndComplexityLog( long epoch );
	class DataLibWriter* OpenComplexityLog( long epoch );
	
	void PickParentsUsingTournament(int numInPool, int* iParent, int* jParent);
	void UpdateAgents();
	void UpdateAgents_StaticTimestepGeometry();

	void Interact();
	void DeathAndStats();
	void MateLockstep();
	int GetMatePotential( agent *x );
	int GetMateStatus( int xPotential, int yPotential );
	int GetMateDenialStatus( agent *x, int *xStatus,
							 agent *y, int *yStatus,
							 short domainID );
	void Mate( agent *c,
			   agent *d,
			   class ContactEntry *contactEntry );
	void Smite( short kd,
				agent *c,
				agent *d );
	int GetFightStatus( agent *x,
						agent *y,
						float *out_power );
	void Fight( agent *c,
				agent *d,
				class ContactEntry *contactEntry,
				bool *cDied,
				bool *dDied );
	int GetGiveStatus( agent *x,
					   Energy &out_energy );
	void Give( agent *x,
			   agent *y,
			   class ContactEntry *contactEntry,
			   bool *xDied,
			   bool toMarkOnDeath );
	void Eat( agent *c,
			  bool *cDied );
	void Carry( agent *c );
	void Pickup( agent *c );
	void Drop( agent *c );
	void Fitness( agent *c );
	void CreateAgents();
	void MaintainBricks();
	void MaintainFood();

	void UpdateMonitors();
	
	void RecordGeneSeparation();
	void CalculateGeneSeparation(agent* ci);
	void CalculateGeneSeparationAll();

	// Following two functions only determine whether or not we should create the relevant files.
	// Linking, renaming, and unlinking are handled according to the specific recording options.
	bool RecordBrainAnatomy( long agentNumber );
	bool RecordBrainFunction( long agentNumber );
	
	void ijfitinc(short* i, short* j);

	void Birth( agent* a,
				LifeSpan::BirthReason reason,
				agent* a_parent1 = NULL,
				agent* a_parent2 = NULL );
 private:
	void Kill( agent* inAgent,
			   LifeSpan::DeathReason reason );
	void Kill_UpdateBrainData( agent *c );
	void Kill_UpdateFittest( agent *c );
	
	void AddFood( long domainNumber, long patchNumber );
	void RemoveFood( food *f );

	void FoodEnergyIn( const Energy &e );
	void FoodEnergyOut( const Energy &e );

	float AgentFitness( agent* c );
	
	void ProcessWorldFile( proplib::Document *docWorldFile );

	void Dump();
	
#if false
	TBinChartWindow* fGeneSeparationWindow;
#endif

	Scheduler fScheduler;
	BusyFetchQueue<agent *> fUpdateBrainQueue;
	
	long fMaxSteps;
	bool fEndOnPopulationCrash;
	int fDumpFrequency;
	bool fLoadState;
	
	gstage fStage;	
	TCastList fWorldCast;

	long fNumDomains;
	
	int fSolidObjects;	// agents cannot pass through solid objects (collisions are avoided)
	int fCarryObjects;  // specifies which types of objects can be picked up.
	int fShieldObjects;  // specifies which types of objects act as shields.

	float fCarryPreventsEat;
	float fCarryPreventsFight;
	float fCarryPreventsGive;
	float fCarryPreventsMate;
	
	float fAgentHealingRate;	// Virgil Healing
	bool fHealing;				// Virgil Healing
	
	float fGroundClearance;
	bool fRecordGeneStats;
	bool fRecordFoodPatchStats;
	bool fCalcFoodPatchAgentCounts;
	
	std::string fComplexityType;
	float fComplexityFitnessWeight;
	float fHeuristicFitnessWeight;

	long fNewLifes;
	long fNewDeaths;
	
	Color fGroundColor;

	agent* fCurrentFittestAgent[MAXFITNESSITEMS];	// based on heuristic fitness
	float fCurrentMaxFitness[MAXFITNESSITEMS];	// based on heuristic fitness
	int fCurrentFittestCount;
	int fNumberFit;
	FitStruct** fFittest;	// based on the complete fitness, however it is being calculated in AgentFitness(c)
	int fNumberRecentFit;
	FitStruct** fRecentFittest;	// based on the complete fitness, however it is being calculated in AgentFitness(c)
	long fFitness1Frequency;
	long fFitness2Frequency;
	short fFitI;
	short fFitJ;
	int fTournamentSize;
	long fPositionSeed;
	long fGenomeSeed;
	long fSimulationSeed;

	float fEatFitnessParameter;
	float fEatThreshold;
	
	float fMaxFitness;
	ulong fNumAverageFitness;
	float fAverageFitness;
	float fPrevAvgFitness;
	float fTotalHeuristicFitness;
	float fMateFitnessParameter;
	float fMoveFitnessParameter;
	float fEnergyFitnessParameter;
	float fAgeFitnessParameter;

	long fMinNumAgents; 
	long fMinNumAgentsWithMetabolism[ MAXMETABOLISMS ];
	long fInitNumAgents;
	long fNumberToSeed;
	float fProbabilityOfMutatingSeeds;
	bool fSeedFromFile;
	std::vector<std::string> fSeedFilePaths;
	bool fPositionSeedsFromFile;
	std::vector<Position> fSeedPositions;
	float fMinMateFraction;
	long fMateWait;
	long fMiscAgents; // number of agents born without intervening creation before miscegenation function kicks in
	float fMateThreshold;
	float fMaxMateVelocity;
	float fMinEatVelocity;
	float fMaxEatVelocity;
	float fMaxEatYaw;
	EatStatistics fEatStatistics;
	long fEatMateSpan;
	float fEatMateMinDistance;
	float fFightThreshold;
	float fFightFraction;
	enum
	{
		FM_NORMAL,
		FM_NULL
	} fFightMode;
	float fGiveThreshold;
	float fGiveFraction;
	float fPickupThreshold;
	float fDropThreshold;
	
	long fNumberAlive;
	long fNumberAliveWithMetabolism[ MAXMETABOLISMS ];
	long fNumberBorn;
	long fNumberBornVirtual;
	long fNumberDied;
	long fNumberDiedAge;
	long fNumberDiedEnergy;
	long fNumberDiedFight;
	long fNumberDiedEat;
	long fNumberDiedEdge;
	long fNumberDiedSmite;
	long fNumberDiedPatch;
	long fNumberCreated;
	long fNumberCreatedRandom;
	long fNumberCreated1Fit;
	long fNumberCreated2Fit;
	long fNumberFights;
	long fBirthDenials;
	long fMiscDenials;
	long fLastCreated;
	long fMaxGapCreate;
	long fNumBornSinceCreated;
	
	bool fUseProbabilisticFoodPatches;
	bool fFoodRemovalNeeded;
	int fNumFoodPatches;
	class FoodPatch* fFoodPatches;
	class FoodPatch** fFoodPatchesNeedingRemoval;

	int fNumBrickPatches;
	class BrickPatch* fBrickPatches;


	long fMinFoodCount;
	long fMaxFoodCount;
	long fMaxFoodGrownCount;
	long fInitFoodCount;
	enum
	{
		RFOOD_FALSE,
		RFOOD_TRUE,
		RFOOD_TRUE__FIGHT_ONLY
	} fAgentsRfood;
	float fFoodRate;
	enum
	{
		MaxRelative,
		MaxIndependent
	} fFoodGrowthModel;
	
	float fFoodRemoveEnergy;
	float fFoodEnergyIn;
	float fFoodEnergyOut;
	float fTotalFoodEnergyIn;
	float fTotalFoodEnergyOut;
	float fAverageFoodEnergyIn;
	float fAverageFoodEnergyOut;
	Energy fEnergyEaten;
	Energy fTotalEnergyEaten;
	float fEat2Consume; // (converts eat neuron value to energy consumed)
	int fFoodPatchOuterRange;
	float fMinFoodEnergyAtDeath;
	float fPower2Energy; // controls amount of damage to other agent
	
	char   fFogFunction;
	float fExpFogDensity;
	int   fLinearFogEnd;

	bool fMonitorGeneSeparation; 	// whether gene-separation will be monitored or not
	bool fRecordGeneSeparation; 	// whether gene-separation will be recorded or not
	float fMaxGeneSeparation;
	float fMinGeneSeparation;
	float fAverageGeneSeparation;
	FILE* fGeneSeparationFile;
	bool fChartGeneSeparation;
	float* fGeneSepVals;
	long fNumGeneSepVals;
	Stat fLifeSpanStats;
	Stat fNeuronGroupCountStats;
	Stat fCurrentNeuronGroupCountStats;
	Stat fCurrentNeuronCountStats;
	Stat fCurrentSynapseCountStats;
	StatRecent fLifeSpanRecentStats;
	StatRecent fLifeFractionRecentStats;

	bool fRandomBirthLocation;
	float fRandomBirthLocationRadius;

	char  fSmiteMode;		// 'R' = Random; 'L' = Leastfit
	float fSmiteFrac;
	float fSmiteAgeFrac;
	int fNumLeastFit;
	int fMaxNumLeastFit;
	int fNumSmited;
	bool fStaticTimestepGeometry;
	bool fParallelInitAgents;
	bool fParallelInteract;
	bool fParallelCreateAgents;
	bool fParallelBrains;
	
	unsigned long* fGeneSum;	// sum, for computing mean
	unsigned long* fGeneSum2;	// sum of squares, for computing std. dev.
	agent** fGeneStatsAgents;   // list of agents to be used for computing stats.
	FILE* fGeneStatsFile;

	FILE* fFoodPatchStatsFile;

	unsigned long fNumAgentsNotInOrNearAnyFoodPatch;
	unsigned long* fNumAgentsInFoodPatch;
	
    gpolyobj fGround;
    TSetList fWorldSet;	

	class AgentPovRenderer *agentPovRenderer;
	class MonitorManager *monitorManager;

	condprop::PropertyList *fConditionalProps;
};

#if false
inline TBinChartWindow* TSimulation::GetGeneSeparationWindow() const { return fGeneSeparationWindow; }
#endif

inline class AgentPovRenderer *TSimulation::GetAgentPovRenderer() { return agentPovRenderer; }
inline MonitorManager *TSimulation::getMonitorManager() { return monitorManager; }
inline gstage &TSimulation::getStage() { return fStage; }


inline bool TSimulation::isLockstep() const { return fLockStepWithBirthsDeathsLog; }
inline long TSimulation::GetMaxAgents() const { return fMaxNumAgents; }
inline long TSimulation::GetInitNumAgents() const { return fInitNumAgents; }
inline long TSimulation::getNumAgents( short domain )
{
	if( domain == -1 )
		return fNumberAlive;
	else
		return fDomains[domain].numAgents;
}

inline long TSimulation::GetNumDomains() const { return fNumDomains; }
inline long TSimulation::getNumBorn( AgentBirthType type )
{
	switch( type )
	{
	case ABT__CREATED: return fNumberCreated;
	case ABT__BORN: return fNumberBorn;
	case ABT__BORN_VIRTUAL: return fNumberBornVirtual;
	default: assert(false); return -1;
	}
}
inline class agent *TSimulation::getCurrentFittest( int rank )
{
	assert( rank != 0 );
	if(rank < 0) {
		if(rank + fCurrentFittestCount >= 0)
			return fCurrentFittestAgent[ rank + fCurrentFittestCount ];
	}
	else if( rank <= fCurrentFittestCount ) return fCurrentFittestAgent[rank - 1];
	return NULL;
}

inline float TSimulation::getFitnessWeight( FitnessWeightType type )
{
	switch( type )
	{
	case FWT__COMPLEXITY: return fComplexityFitnessWeight;
	case FWT__HEURISTIC: return fHeuristicFitnessWeight;
	default: assert(false); return -1;
	}
}

inline float TSimulation::getFitnessStat( FitnessStatType type )
{
	switch( type )
	{
	case FST__MAX_FITNESS: return fMaxFitness;
	case FST__CURRENT_MAX_FITNESS: return fCurrentMaxFitness[0];
	case FST__AVERAGE_FITNESS: return fAverageFitness;
	default: assert(false); return -1;
	}
}
inline float TSimulation::getFoodEnergyStat( FoodEnergyStatType type, FoodEnergyStatScope scope )
{
	switch( type )
	{
	case FEST__IN:
		switch(scope)
		{
		case FESS__STEP: return fFoodEnergyIn;
		case FESS__TOTAL: return fTotalFoodEnergyIn;
		case FESS__AVERAGE: return fAverageFoodEnergyIn;
		default: assert(false); return -1;
		}
		break;
	case FEST__OUT:
		switch(scope)
		{
		case FESS__STEP: return fFoodEnergyOut;
		case FESS__TOTAL: return fTotalFoodEnergyOut;
		case FESS__AVERAGE: return fAverageFoodEnergyOut;
		default: assert(false); return -1;
		}
		break;
	default: assert(false); return -1;
	}
}
inline long TSimulation::getStep() const { return fStep; }
inline long TSimulation::GetMaxSteps() const { return fMaxSteps; }
inline float TSimulation::EnergyFitnessParameter() const { return fEnergyFitnessParameter; }
inline float TSimulation::AgeFitnessParameter() const { return fAgeFitnessParameter; }
inline float TSimulation::LifeFractionRecent() { return fLifeFractionRecentStats.mean(); }
inline unsigned long TSimulation::LifeFractionSamples() { return fLifeFractionRecentStats.samples(); }


inline float TSimulation::GetAgentHealingRate() { return fAgentHealingRate;	} 


// Following two functions only determine whether or not we should create the relevant files.
// Linking, renaming, and unlinking are handled according to the specific recording options.
inline bool TSimulation::RecordBrainAnatomy( long agentNumber )
{
	return( fBestSoFarBrainAnatomyRecordFrequency ||
			fBestRecentBrainAnatomyRecordFrequency ||
			fBrainAnatomyRecordAll ||
			(fBrainAnatomyRecordSeeds && (agentNumber <= fInitNumAgents))
		  );
}
inline bool TSimulation::RecordBrainFunction( long agentNumber )
{
	return( fBestSoFarBrainFunctionRecordFrequency ||
			fBestRecentBrainFunctionRecordFrequency ||
			fBrainFunctionRecentRecordFrequency ||
			fBrainFunctionRecordAll ||
			(fBrainFunctionRecordSeeds && (agentNumber <= fInitNumAgents))
		  );
}

