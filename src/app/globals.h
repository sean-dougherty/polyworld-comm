/********************************************************************/
/* PolyWorld:  An Artificial Life Ecological Simulator              */
/* by Larry Yaeger                                                  */
/* Copyright Apple Computer 1990,1991,1992                          */
/********************************************************************/
// globals.h - global variables for pw (PolyWorld)

#ifndef GLOBALS_H
#define GLOBALS_H

// System
#include <stdio.h>

// STL
#include <map>
#include <string>

// Local
#include "AbstractFile.h"

// Forward declarations
class agent;

#define MAXLIGHTS 10

class globals
{
public:
	static float	worldsize;
	static bool		wraparound;
	static bool		blockedEdges;
	static bool     stickyEdges;
	static int      numEnergyTypes;
    static int      numSoundFrequencies;
	static AbstractFile::ConcreteFileType recordFileType;
};

#endif
