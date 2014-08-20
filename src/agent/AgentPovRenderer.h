#pragma once

#include <map>

#include "AgentAttachedData.h"

class AgentPovRenderer
{
 public:
	AgentPovRenderer( int maxAgents,
					  int retinaWidth,
					  int retinaHeight );
	virtual ~AgentPovRenderer();

	void add( class agent *a );
	void remove( class agent *a );

	void beginStep();
	void render( class agent *a );
	void endStep();

	int getBufferWidth();
	int getBufferHeight();

 private:
	struct Viewport
	{
		int index;
		short x;
		short y;
		short width;
		short height;
	};
	int fBufferWidth;
	int fBufferHeight;
	// This gives us a reference to a per-agent opaque pointer.
	AgentAttachedData::SlotHandle slotHandle;
	Viewport *fViewports;
	std::map<int, Viewport *> fFreeViewports;
};
