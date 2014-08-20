#pragma once

#include "Sensor.h"

#include <vector>

class agent;
class Nerve;
class NervousSystem;

class HearingSensor : public Sensor
{
 public:
    static const int ReceptorCount = 3;

	HearingSensor( agent *self );
	virtual ~HearingSensor();

	virtual void sensor_grow( NervousSystem *cns );
	virtual void sensor_prebirth_signal( RandomNumberGenerator *rng );
	virtual void sensor_update( bool print );

    void add_sound(float intensity, int frequency, float x, float z);

 private:
	agent *self;
    Nerve *nerve;
    struct Sound {
        float angle;
        float distance;
        float intensity;
        int frequency;
    };
    std::vector<Sound> sounds;

    class Receptor {
    public:
        void init(agent *self, Nerve *nerve, int index, float angle);

        void update(std::vector<Sound> &sounds);

        agent *self;
        Nerve *nerve;
        int index;
        float angle;
    } receptors[ReceptorCount];
};
