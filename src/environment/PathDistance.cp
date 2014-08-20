#include "PathDistance.h"

#include <cassert>
#include <functional>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <map>
using namespace std;

#define errif(cond) {if(cond) {fprintf(stderr, "%s:%d: Failing condition (%s)\n", __FILE__, __LINE__, #cond); exit(1);}}
#define erriff(cond,msg...) {if(cond) {fprintf(stderr, "%s:%d: Failing condition (%s)\n", __FILE__, __LINE__, #cond); fprintf(stderr, msg); fprintf(stderr, "\n"); exit(1);}}

static float dist(float x1, float y1, float x2, float y2) {
    float dx = x1-x2;
    float dy = y1-y2;
    return sqrt( dx*dx + dy*dy );
}

static bool same(float x1, float y1, float x2, float y2) {
    return dist(x1,y1, x2,y2) < 0.01;
}

struct Point {
    float x;
    float y;
};

struct Segment {
    float x1;
    float y1;
    float x2;
    float y2;

    static int idcount;
    int id;

    float a, b, c;
    float xmin, xmax, ymin, ymax;

    vector<Segment*> connections;

    map<int, float> distmap1, distmap2;

    static size_t N() {
        return idcount;
    }

    Segment(float x1_, float y1_, float x2_, float y2_) {
        x1 = x1_;
        y1 = y1_;
        x2 = x2_;
        y2 = y2_;
        id = idcount++;
        a = y2 - y1;
        b = x1 - x2;
        c = x2 * y1 - x1 * y2;
        xmin = min(x1, x2) - 0.01;
        xmax = max(x1, x2) + 0.01;
        ymin = min(y1, y2) - 0.01;
        ymax = max(y1, y2) + 0.01;
    }

    float length() const {
        return ::dist(x1,y1,x2,y2);
    }

    float dist(Point p) {
        return fabs(a*p.x + b*p.y + c) / sqrt(a*a + b*b);
    }

    Point projection(Point p) {
        Point result = {
            (b*(b*p.x - a*p.y) - a*c) / (a*a + b*b),
            (a*(-b*p.x + a*p.y) - b*c) / (a*a + b*b) };
                
        return result;
    }

    float contains_projection(Point p) {
        return p.x >= xmin && p.x <= xmax && p.y >= ymin && p.y <= ymax;
    }

    bool connects(const Segment &other) {
        return
            same(x1, y1, other.x1, other.y1) ||
            same(x1, y1, other.x2, other.y2) ||
            same(x2, y2, other.x1, other.y1) ||
            same(x2, y2, other.x2, other.y2);
    }

    void connect(Segment &other) {
        connections.push_back(&other);
        other.connections.push_back(this);
    }

    void __update_dist(int id, float accum, Segment *sibling) {
        if(same(x1, y1, sibling->x1, sibling->y1) || same(x1, y1, sibling->x2, sibling->y2)) {
            if( (distmap1.find(id) == distmap1.end()) || (distmap1[id] > accum) )
                distmap1[id] = accum;
            else
                return;
        } else {
            if( (distmap2.find(id) == distmap2.end()) || (distmap2[id] > accum) )
                distmap2[id] = accum;
            else
                return;
        }

        accum += length();

        for(auto pseg: connections) {
            if(pseg != sibling)
                pseg->__update_dist(id, accum, this);
        }
    }

    void update_dist() {
        distmap1[id] = 0.0;
        distmap2[id] = 0.0;
        for(auto pseg: connections) {
            pseg->__update_dist(id, 0.0, this);
        }
    }

    void prune_dist() {
        for(auto pair: distmap1) {
            auto it = distmap2.find(pair.first);
            if( (it != distmap2.end()) && (it->second > pair.second) ) {
                distmap2.erase(it);
            }
        }

        for(auto pair: distmap2) {
            auto it = distmap1.find(pair.first);
            if( (it != distmap1.end()) && (it->second > pair.second) ) {
                distmap1.erase(it);
            }
        }
    }
};

static vector<Segment> segments;

int Segment::idcount = 0;

static ostream &operator<<(ostream &out, const Segment &seg) {
    return cout << "(" << seg.x1 << "," << seg.y1 << "," << seg.x2 << "," << seg.y2 << ") --> " << seg.length() << ", [" << seg.a << "," << seg.b << "," << seg.c << "]";
}

static ostream &operator<<(ostream &out, const Point &p) {
    return cout << "(" << p.x << "," << p.y << ")";
}

static void dump_segments() {
    for(auto seg: segments) {
        cout << "[" << seg.id << "]: " << seg << endl;
    }
}

static void dump_distmaps() {
    for(auto &seg: segments) {
        cout << "[" << seg.id << "]: " << seg << endl;
        cout << "d1:" << endl;
        for(auto pair: seg.distmap1) {
            cout << "  " << pair.first << ": " << pair.second << endl;
        }
        cout << "d2:" << endl;
        for(auto pair: seg.distmap2) {
            cout << "  " << pair.first << ": " << pair.second << endl;
        }
    }
}

static void dump_connections() {
    for(auto &seg: segments) {
        cout << "[" << seg.id << "]: " << seg << endl;
        for(auto pseg: seg.connections) {
            cout << "  [" << pseg->id << "]: " << *pseg << endl;
        }
    }
}

static void connectall() {
    for(size_t i = 0; i < Segment::N(); i++) {
        for(size_t j = i+1; j < Segment::N(); j++) {
            if(segments[i].connects(segments[j]))
                segments[i].connect(segments[j]);
        }
    }

    for(auto &seg: segments) {
        errif(seg.connections.empty());
    }
}

static Segment *get_segment(Point p) {
    Segment *result = NULL;
    float mindist;

    for(auto &seg: segments) {
        Point proj = seg.projection(p);
        if(seg.contains_projection(proj)) {
            float dist = seg.dist(p);
            if(result == NULL || dist < mindist) {
                result = &seg;
                mindist = dist;
            }
        }
    }

    return result;
}

static float get_intra_segment_distance(Point p, Segment *src_seg, Segment *dst_seg) {
    if(src_seg->distmap1.find(dst_seg->id) != src_seg->distmap1.end()) {
        return dist(p.x,p.y, src_seg->x1, src_seg->y1) + src_seg->dist(p);
    } else {
        return dist(p.x,p.y, src_seg->x2, src_seg->y2) + src_seg->dist(p);
    }
}

static float get_inter_segment_distance(Segment *src_seg, Segment *dst_seg) {
    if(src_seg->distmap1.find(dst_seg->id) != src_seg->distmap1.end()) {
        return src_seg->distmap1[dst_seg->id];
    } else {
        return src_seg->distmap2[dst_seg->id];
    }
}

static float path_dist(Point p1, Point p2) {
    Segment *seg1 = get_segment(p1);
    erriff(!seg1, "No segment at (%f, %f)", p1.x, p1.y);

    Segment *seg2 = get_segment(p2);
    erriff(!seg2, "No segment at (%f, %f)", p2.x, p2.y);

     
    if(seg1 == seg2) {
        return dist(p1.x,p1.y, p2.x,p2.y);
    } else {
        return get_intra_segment_distance(p1, seg1, seg2)
            + get_intra_segment_distance(p2, seg2, seg1)
            + get_inter_segment_distance(seg1, seg2);
    }
}


bool PathDistance::_complete = false;

void PathDistance::addSegment(float x1, float y1, float x2, float y2) {
    errif(_complete);

    segments.push_back( Segment(x1, y1, x2, y2) );
}

void PathDistance::complete() {
    const bool debug = false;

    errif(_complete);
    _complete = true;

    if(debug) dump_segments();

    connectall();
    if(debug) dump_connections();

    for(auto &seg: segments) {
        seg.update_dist();
    }
    for(auto &seg: segments) {
        seg.prune_dist();
    }
    if(debug) dump_distmaps();

    for(auto &seg: segments) {
        errif( (seg.distmap1.size() + seg.distmap2.size()) != (segments.size() + 1) );
    }
}

float PathDistance::distance(float x1, float y1, float x2, float y2) {
    errif(!_complete);

    return path_dist( {x1, y1}, {x2, y2} );
}

unsigned PathDistance::getSegmentCount() {
    return segments.size();
}

int PathDistance::getSegmentId(float x, float y) {
    return get_segment({x,y})->id;
}

void PathDistance::test() {
    PathDistance::addSegment(0.0, -136.93, 0.0, 0.0);
    PathDistance::addSegment(0.0, 0.0, 0.0, 33.06);
    PathDistance::addSegment(0.0, 33.06, -11.97, 43.87);
    PathDistance::addSegment(-11.97, 43.87, -20.67, 43.05);
    PathDistance::addSegment(-11.97, 43.87, -20.38, 52.40);
    PathDistance::addSegment(-11.97, 43.87, -10.80, 52.79);
    PathDistance::addSegment(0.0, 33.06, 0.0, 64.28);
    PathDistance::addSegment(0.0, 64.28, -6.93, 70.04);
    PathDistance::addSegment(0.0, 64.28, 0.0, 78.61);

    PathDistance::complete();

    struct {
        Point a;
        Point f;
        float d;
    } tests[] = {
        {{-0.59, -13.42}, {-10.45, 42.36}, 60.482},
        {{-4.8, 35.95}, {-6.29, 69.86}, 45.2312},
        {{-15.14, 47.34}, {-19.54, 51.66}, 6.16623}
    };

    for(auto &t: tests) {
        float dist = PathDistance::distance(t.a.x, t.a.y, t.f.x, t.f.y);
        cout << t.a << ", " << t.f << " --> " << dist << endl;
        errif( t.d >= 0.0 && fabs(dist - t.d) > 0.01);
    }
}
