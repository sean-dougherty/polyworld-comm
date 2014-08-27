// System
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/stat.h>

// STL
#include <string>

// Local
#include "pwmpi.h"
#include "Resources.h"
#include "Simulation.h"

//#define UNIT_TESTS

#ifdef UNIT_TESTS
#include "PathDistance.h"

static void unit_tests()
{
    PathDistance::test();
}
#endif

using namespace std;

//===========================================================================
// usage
//===========================================================================
void usage( const char* format, ... )
{
	printf( "Usage:  Polyworld [-o <rundir>] <worldfile>\n" );
	
	if( format )
	{
		printf( "Error:\n\t" );
		va_list argv;
		va_start( argv, format );
		vprintf( format, argv );
		va_end( argv );
		printf( "\n" );
	}
	
	exit( 1 );
}

void usage()
{
	usage( NULL );
}

//===========================================================================
// main
//===========================================================================
int main( int argc, char** argv )
{
#ifdef UNIT_TESTS
    unit_tests();
    exit(0);
#endif

    Resources::init();

    int nargs = 0;
	string worldfilePath;
    string rundir = "run";
	
	for( int argi = 1; argi < argc; argi++ ) {
		string arg = argv[argi];

		if( arg[0] == '-' ) {
			if( arg == "-o" ) {
				if( ++argi >= argc )
					usage( "Missing -o arg" );
				rundir = argv[argi];
            } else {
				usage( "Unknown argument: %s", argv[argi] );
            }
		} else {
            nargs++;
            worldfilePath = Resources::get_user_path(argv[argi]);
		}
	}

    if(nargs != 1) {
        usage( "Expecting one worldfile as argument" );
    }

    pwmpi::init(&argc, &argv);
    if( pwmpi::size() > 1) {
        char path[1024];
        sprintf(path, "%s_rank%d", rundir.c_str(), pwmpi::rank());
        rundir = path;
    }

	// ---
	// --- Create the run directory and cd into it
	// ---
	{
		char rundir_saved[1024];

		// First save the old directory, if it exists
		sprintf( rundir_saved, "%s_%ld", rundir.c_str(), time(NULL) );
		(void) rename( rundir.c_str(), rundir_saved );

		if( mkdir(rundir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) )
		{
			eprintf( "Error making run directory %s (%d)\n", rundir.c_str(), errno );
			exit( 1 );
		}

        require( 0 == chdir(rundir.c_str()) );
	}

	TSimulation *simulation = new TSimulation( worldfilePath );

    while(!simulation->isEnded()) {
        simulation->Step();
    }

	delete simulation;

    pwmpi::finalize();

	return 0;
}