#pragma once

#include <stdio.h>
#include <stdlib.h>

#define errif( STMT, MSG... ) if( STMT ) { fprintf(stderr, "[%s:%d] '%s' ", __FILE__, __LINE__, #STMT); fprintf(stderr, MSG); fprintf(stderr, "\n"); exit(1); }
#define require( STMT ) if( !(STMT) ) { fprintf(stderr, "ASSERTION ERROR! [%s:%d] '%s'\n", __FILE__, __LINE__, #STMT); exit(1); }
#define panic() { fprintf(stderr, "PANIC! [%s:%d]\n", __FILE__, __LINE__); exit(1); }
