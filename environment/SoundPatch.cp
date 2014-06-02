#include "SoundPatch.h"
#include "objectxsortedlist.h"

#include <iostream>
using namespace std;

void SoundPatch::init(float centerX, float centerZ, float sizeX, float sizeZ, std::vector<int> sequence_)
{
    startX = centerX - (sizeX / 2);
    endX = centerX + (sizeX / 2);
    startZ = centerZ - (sizeZ / 2);
    endZ = centerZ + (sizeZ / 2);
    index = 0;

    for(int i: sequence_)
    {
        for(int j = 0; j < 3; j++)
            sequence.push_back(-1);

        for(int j = 0; j < 5; j++)
            sequence.push_back(i);
    }
}

void SoundPatch::update(long step)
{
    if(index >= sequence.size())
        return;

    int frequency = sequence[index++];
    if(frequency < 0)
        return;

    int nagents = 0;
    cout << "sound @ " << step << ": (+1)" << (frequency+1) << endl;

    gdlink<gobject*> *saveCurr = objectxsortedlist::gXSortedObjects.getcurr();

    agent *a;
    objectxsortedlist::gXSortedObjects.reset();
    while( objectxsortedlist::gXSortedObjects.nextObj( AGENTTYPE, (gobject**) &a ) )
    {
        if( a->x() >= startX && a->x() <= endX && a->z() >= startZ && a->z() <= endZ )
        {
            a->sound(1.0, frequency, a->x(), a->z());
            nagents++;
        }
    }
    objectxsortedlist::gXSortedObjects.setcurr( saveCurr );

    cout << "  sent to " << nagents << " agents" << endl;
}
