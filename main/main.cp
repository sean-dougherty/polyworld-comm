// System
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/stat.h>

// STL
#include <string>

// Qt
#include <qgl.h>
#include <QApplication>

// Local
#include "FiringRateModel_Cuda.h"
#include "MainWindow.h"
#include "Monitor.h"
#include "pwmpi.h"
#include "Resources.h"
#include "Simulation.h"
#include "SimulationController.h"
#include "TerminalUI.h"

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
	printf( "Usage:  Polyworld [--ui gui|term] [-o <rundir>] [--mf <monitor_file>] <worldfile>\n" );
	
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
	string ui = "term";
    string rundir = "run";
	string monitorPath;
	
	for( int argi = 1; argi < argc; argi++ ) {
		string arg = argv[argi];

		if( arg[0] == '-' ) {
			if( arg == "--ui" ) {
				if( ++argi >= argc )
					usage( "Missing --ui arg" );
			
				ui = argv[argi];
				if( (ui != "gui") && (ui != "term") )
					usage( "Invalid --ui arg (%s)", argv[argi] );
            } else if( arg == "--mf" ) {
				if( ++argi >= argc )
					usage( "Missing --mf arg" );
			
				monitorPath = Resources::get_user_path(argv[argi]);
			} else if( arg == "-o" ) {
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

    if(monitorPath.empty()) {
        monitorPath = Resources::get_user_path("./" + ui + ".mf");
		if( !exists(monitorPath) )
			monitorPath = Resources::get_pw_path("./etc/" + ui + ".mf");
	}

    //pwmpi::init(&argc, &argv);
    if(pwmpi::is_mpi_mode() && !pwmpi::is_master()) {
        FiringRateModel_Cuda::config(pwmpi::get_gpu_index());
    }

    cout << "wf=" << worldfilePath << ", mf=" << monitorPath << endl;

	QApplication app(argc, argv);

    if (!QGLFormat::hasOpenGL())
    {
		qWarning("This system has no OpenGL support. Exiting.");
		return -1;
    }	

	// Establish how our preference settings file will be named
	QCoreApplication::setOrganizationDomain( "indiana.edu" );
	QCoreApplication::setApplicationName( "polyworld" );

	// ---
	// --- Create the run directory and cd into it
	// ---
	{
		char rundir_saved[1024];

		// First save the old directory, if it exists
		sprintf( rundir_saved, "%s_%ld", rundir.c_str(), time(NULL) );
		(void) rename( rundir.c_str(), rundir_saved );

// Define directory mode mask the same, except you need execute privileges to use as a directory (go fig)
#define	PwDirMode ( S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH )

		if( mkdir(rundir.c_str(), PwDirMode) )
		{
			eprintf( "Error making run directory %s (%d)\n", rundir.c_str(), errno );
			exit( 1 );
		}

        require( 0 == chdir(rundir.c_str()) );
	}

	TSimulation *simulation = new TSimulation( worldfilePath, monitorPath );
	SimulationController *simulationController = new SimulationController( simulation );

	int exitval;

	if( ui == "gui" )
	{
		MainWindow *mainWindow = new MainWindow( simulationController );
		simulationController->start();
		exitval = app.exec();
		delete mainWindow;
	}
	else if( ui == "term" )
	{
		TerminalUI *term = new TerminalUI( simulationController );
		simulationController->start();
		exitval = app.exec();
		delete term;
	}
	else
		panic();

	delete simulationController;
	delete simulation;

	return exitval;
}
