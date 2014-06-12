#pragma once

#include <vector>

class SoundPatch
{
public:
    void init(float centerX, float centerZ, float sizeX, float sizeZ, std::vector<int> sequence);

    void activate(long step);
    void update(long step);

private:
    float startX;
    float endX;
    float startZ;
    float endZ;
    std::size_t index;
    std::vector<int> sequence;
};
