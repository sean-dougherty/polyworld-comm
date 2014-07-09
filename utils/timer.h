#pragma once

#include <sys/time.h>

static double seconds() {
	struct timeval tv;
	gettimeofday( &tv, NULL );

	return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}
