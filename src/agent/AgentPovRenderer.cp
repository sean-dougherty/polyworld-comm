#include "AgentPovRenderer.h"

#include "agent.h"
#include "pwassert.h"
#include "Retina.h"

#define CELL_PAD 2

//---------------------------------------------------------------------------
// AgentPovRenderer::AgentPovRenderer
//---------------------------------------------------------------------------
AgentPovRenderer::AgentPovRenderer( int maxAgents,
									int retinaWidth,
									int retinaHeight )
{
	// If we decide we want the width W (in cells) to be a multiple of N (call it I)
	// and we want the aspect ratio of W to height H (in cells) to be at least A,
	// and we call maxAgents M, then (do the math or trust me):
	// I = floor( (sqrt(M*A) + (N-1)) / N )
	// W = I * N
	// H = floor( (M + W - 1) / W )
	int n = 10;
	int a = 3;
	int i = (int) (sqrt( (float) (maxAgents * a) ) + n - 1) / n;
	int ncols = i * n;   // width in cells
	int nrows = (maxAgents + ncols - 1) / ncols;
	fBufferWidth = ncols * (retinaWidth + CELL_PAD);
	fBufferHeight = nrows * (retinaHeight + CELL_PAD);

	slotHandle = AgentAttachedData::createSlot();

	fViewports = new Viewport[ maxAgents ];
	for( int i = 0; i < maxAgents; i++ )
	{
		Viewport *viewport = fViewports + i;

		viewport->index = i;
		
		short irow = short(i / ncols);            
		short icol = short(i) - (ncols * irow);

		viewport->width = retinaWidth;
		viewport->height = retinaHeight;

		viewport->x = icol * (viewport->width + CELL_PAD)  +  CELL_PAD;
		short ytop = fBufferHeight  -  (irow) * (viewport->height + CELL_PAD) - CELL_PAD - 1;
		viewport->y = ytop  -  viewport->height  +  1;

		fFreeViewports.insert( make_pair(viewport->index, viewport) );
	}
}

//---------------------------------------------------------------------------
// AgentPovRenderer::~AgentPovRenderer
//---------------------------------------------------------------------------
AgentPovRenderer::~AgentPovRenderer()
{
	delete [] fViewports;
}

//---------------------------------------------------------------------------
// AgentPovRenderer::add
//---------------------------------------------------------------------------
void AgentPovRenderer::add( agent *a )
{
	assert( AgentAttachedData::get( a, slotHandle ) == NULL );

	Viewport *viewport = fFreeViewports.begin()->second;
	fFreeViewports.erase( fFreeViewports.begin() );

	AgentAttachedData::set( a, slotHandle, viewport );
}

//---------------------------------------------------------------------------
// AgentPovRenderer::remove
//---------------------------------------------------------------------------
void AgentPovRenderer::remove( agent *a )
{
	Viewport *viewport = (Viewport *)AgentAttachedData::get( a, slotHandle );

	if( viewport )
	{
		AgentAttachedData::set( a, slotHandle, NULL );
		fFreeViewports.insert( make_pair(viewport->index, viewport) );
	}
}

//---------------------------------------------------------------------------
// AgentPovRenderer::beginStep
//---------------------------------------------------------------------------
void AgentPovRenderer::beginStep()
{
}

//---------------------------------------------------------------------------
// AgentPovRenderer::render
//---------------------------------------------------------------------------
void AgentPovRenderer::render( agent *a )
{
    panic();
}

//---------------------------------------------------------------------------
// AgentPovRenderer::endStep
//---------------------------------------------------------------------------
void AgentPovRenderer::endStep()
{
}

//---------------------------------------------------------------------------
// AgentPovRenderer::getBufferWidth
//---------------------------------------------------------------------------
int AgentPovRenderer::getBufferWidth()
{
	return fBufferWidth;
}

//---------------------------------------------------------------------------
// AgentPovRenderer::getBufferHeight
//---------------------------------------------------------------------------
int AgentPovRenderer::getBufferHeight()
{
	return fBufferHeight;
}
