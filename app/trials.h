#pragma once

#define TRIALS true

#if TRIALS
#include "FittestList.h"

#include <map>
#include <vector>

#define TEST_INTERLUDE 10
#define NTRIALS 10

class agent;
class TSimulation;
namespace genome {
    class Genome;
}

struct Test
{
    virtual ~Test() {}

    virtual long get_trial_timestep_count() = 0;

    virtual void timestep_input(int trial_number,
                                long test_timestep,
                                agent *a,
                                int freq) = 0;
    virtual void timestep_output(int trial_number,
                                 long test_timestep,
                                 agent *a,
                                 int freq) = 0;

    virtual void log_performance(agent *a,
                                 const char *path_dir) = 0;

    virtual void reset() = 0;
};

struct TrialsState
{
    TrialsState(TSimulation *sim_);
    ~TrialsState();

    void timestep_begin();
    void timestep_end();

    TSimulation *sim;
    std::vector<int> freq_sequence;

    std::vector<Test *> tests;

    long test_number;
    long trial_number;
    long trial_timestep;
    long trial_end_sim_step;
    
private:
    long generation_number;
    std::vector<agent *> generation_agents;
    FittestList global_elites;
    FittestList generation_elites;

    vector<agent *> create_generation(size_t nagents,
                                      size_t nseeds,
                                      size_t ncrossover);
    void init_generation_genomes(vector<agent *> &agents,
                                 size_t nseeds,
                                 size_t ncrossover);

    void new_test();
    void new_trial();
    void new_generation();
    void end_generation();

    void log_elite(agent *a, float score);
};

extern TrialsState *trials;
#endif
