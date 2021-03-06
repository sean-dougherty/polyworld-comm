#pragma once

class PathDistance
{
public:
    static void addSegment(float x1, float y1, float x2, float y2);
    static void complete();
    static float distance(float x1, float y1, float x2, float y2);
    static unsigned getSegmentCount();
    static int getSegmentId(float x, float y);

    static void test();
private:
    static bool _complete;
};
