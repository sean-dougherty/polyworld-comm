#include "HearingSensor.h"

#include <assert.h>

#include "agent.h"
#include "NervousSystem.h"

HearingSensor::HearingSensor( agent *self )
{
	this->self = self;
}

HearingSensor::~HearingSensor()
{
}

void HearingSensor::sensor_grow( NervousSystem *cns )
{
	nerve = cns->getNerve( "Hearing" );

    receptors[0].init(self, nerve, 0, 60);
    receptors[1].init(self, nerve, 1, 180);
    receptors[2].init(self, nerve, 2, 300);
}

void HearingSensor::sensor_prebirth_signal( RandomNumberGenerator *rng )
{
    for( int i = 0; i < nerve->getNeuronCount(); i++ )
        nerve->set( i, 0.0 );
}

void HearingSensor::sensor_update( bool print )
{
    for(int i = 0; i < ReceptorCount; i++)
    {
        receptors[i].update(sounds);
    }

    sounds.clear();
}

void HearingSensor::add_sound(float intensity, int frequency, float x, float z)
{
    float dx = x - self->x();
    float dz = z - self->z();
    float distance = max( 1.0f, sqrtf(dx*dx + dz*dz) );
    float angle = atan2f(-dx, -dz) * RADTODEG - self->yaw();

    sounds.push_back({angle, distance, intensity, frequency});
}

void HearingSensor::Receptor::init(agent *self_, Nerve *nerve_, int index_, float angle_)
{
    self = self_;
    nerve = nerve_;
    index = index_;
    angle = angle_;
}

void HearingSensor::Receptor::update(std::vector<Sound> &sounds) {
    float total_intensity[globals::numSoundFrequencies];
    for(int i = 0; i < globals::numSoundFrequencies; i++)
        total_intensity[i] = 0.0f;

    for(auto &sound: sounds)
    {
        float angle = sound.angle - this->angle;
        angle *= DEGTORAD;

        // Scale intensity based on angle to receptor. If the sound is directly in line
        // with this receptor, then the scale is 1.0. If it is directly behind, then
        // the scale is 0.1.
        float angle_scale = 0.1f + 0.9f*(0.5f + 0.5f * cosf(angle));

        // Scale intensity by 1/r
        float distance_scale = 1.0f / sound.distance;

        float scaled_intensity = sound.intensity * angle_scale * distance_scale;

        total_intensity[sound.frequency] += scaled_intensity;
    }

    int neuron_index = this->index * globals::numSoundFrequencies;

    for(int i = 0; i < globals::numSoundFrequencies; i++) {
        nerve->set( neuron_index + i,
                    min(1.0f, total_intensity[i]) );
    }
}
