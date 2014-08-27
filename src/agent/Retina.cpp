#include "Retina.h"

#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>

#include "AbstractFile.h"
#include "Brain.h"
#include "NervousSystem.h"
#include "RandomNumberGenerator.h"
#include "Simulation.h"

// SlowVision, if turned on, will cause the vision neurons to slowly
// integrate what is rendered in front of them into their input neural activation.
// They will do so at a rate defined by TauVision, adjusting color like this:
//     new = TauVision * image  +  (1.0 - TauVision) * old
// SlowVision is traditionally off, and TauVision will initially be 0.2.
#define SlowVision false
#define TauVision 0.2


Retina::Retina( int width )
{
	this->width = width;
	
	buf = (unsigned char *)calloc( width * 4, sizeof(unsigned char) );

#if PrintBrain
	bprinted = false;
#endif
}

Retina::~Retina()
{
	free( buf );
}

void Retina::force_color(float r, float g, float b) {
    channels[0].forced_value = r;
    channels[1].forced_value = g;
    channels[2].forced_value = b;
}

void Retina::sensor_grow( NervousSystem *cns )
{
	channels[0].init( this, cns, 0, "Red" );
	channels[1].init( this, cns, 1, "Green" );
	channels[2].init( this, cns, 2, "Blue" );
}

void Retina::sensor_prebirth_signal( RandomNumberGenerator *rng )
{
	for( int i = 0; i < width * 4; i++ )
	{
		buf[i] = (unsigned char)(rng->range(0.0, 255.0));
	}

	sensor_update( false );
}

void Retina::sensor_update( bool bprint )
{
	IF_BPRINTED
	(
		printf( "numneurons red,green,blue=%d, %d, %d\n",
				channels[0].numneurons, channels[1].numneurons, channels[2].numneurons );
		printf( "xwidth red,green,blue=%g, %g, %g\n",
				channels[0].xwidth, channels[1].xwidth, channels[2].xwidth );
		printf( "xintwidth red,green,blue=%d, %d, %d,\n",
				channels[0].xintwidth, channels[1].xintwidth, channels[2].xintwidth );
	)

	for( int i = 0; i < 3; i++ )
	{
		channels[i].update( bprint );
	}

	IF_BPRINT
	(
        printf("***** step = %ld ******\n", TSimulation::fStep);
        printf("retinaBuf [0 - %d]\n",(Brain::config.retinaWidth - 1));
        printf("red:");
        
        for( int i = 0; i < (Brain::config.retinaWidth * 4); i+=4 )
            printf(" %3d", buf[i]);
        printf("\ngreen:");
        
        for( int i = 1; i < (Brain::config.retinaWidth * 4); i+=4 )
            printf(" %3d",buf[i]);
        printf("\nblue:");
        
        for( int i = 2; i < (Brain::config.retinaWidth * 4); i+=4 )
            printf(" %3d", buf[i]);
        printf("\n");		
	)
}

void Retina::sensor_start_functional( AbstractFile *f )
{
	for( int i = 0; i < 3; i++ )
	{
		channels[i].start_functional( f );
	}
}

void Retina::sensor_dump_anatomical( AbstractFile *f )
{
	for( int i = 0; i < 3; i++ )
	{
		channels[i].dump_anatomical( f );
	}
}

void Retina::updateBuffer( short x, short y,
						   short width, short height )
{
    panic();
	// The POV window must be the current GL context,
	// when agent::UpdateVision is called, for both the
	// DrawAgentPOV() call above and the glReadPixels()
	// call below.  It is set in TSimulation::Step().
			
	glReadPixels(x,
				 y + height / 2,
				 width,
				 1,
				 GL_RGBA,
				 GL_UNSIGNED_BYTE,
				 buf);

	debugcheck( "after glReadPixels" );
}

const unsigned char *Retina::getBuffer()
{
	return buf;
}

void Retina::Channel::init( Retina *retina,
							NervousSystem *cns,
							int index,
							const char *name )
{
	this->name = name;
	this->index = index;
	buf = retina->buf;

	nerve = cns->getNerve( name );

	numneurons = nerve->getNeuronCount();

	int width = retina->width;

	if( numneurons > 0 )
	{
		xwidth = float(width) / numneurons;
		xintwidth = width / numneurons;

		if( (xintwidth * numneurons) != width )
		{
			xintwidth = 0;
		}
	}
}

void Retina::Channel::update( bool bprint )
{
    for(int i = 0; i < numneurons; i++) {
        nerve->set(i, forced_value);
    }
}

void Retina::Channel::start_functional( AbstractFile *f )
{
	int i = nerve->getIndex();

	f->printf( " %d-%d", i, i + numneurons - 1 );
}

void Retina::Channel::dump_anatomical( AbstractFile *f )
{
	int i = nerve->getIndex();

	char name[128];
	sprintf( name, "%c%sinput", tolower(nerve->name[0]), nerve->name.substr(1).c_str() );

	f->printf( " %s=%d-%d", name, i, i + numneurons - 1 );
}
