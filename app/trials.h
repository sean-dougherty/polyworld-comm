#pragma once

#define TRIALS true

#if TRIALS
#include "FittestList.h"

#include <map>
#include <vector>
#include <random>

#define TEST_INTERLUDE 10
#define NTRIALS 6

class agent;
class TSimulation;
namespace genome {
    class Genome;
}

struct Test
{
    virtual ~Test() {}

    virtual void init(size_t ntrials, size_t nagents) = 0;

    virtual long get_trial_timestep_count() = 0;

    virtual void timestep_input(int trial_number,
                                long test_timestep,
                                agent *a,
                                int freq) = 0;
    virtual void timestep_output(int trial_number,
                                 long test_timestep,
                                 agent *a,
                                 int freq) = 0;
};

struct Deme {
    Deme(TSimulation *sim_, size_t id_, size_t nagents_, size_t nelites);

    vector<agent *> create_generation(long generation_number_);
    void init_generation0_genomes(vector<agent *> &agents);
    void init_generation_genomes(vector<agent *> &next_generation);
    void end_generation();

    FitStruct *get_fittest();
    void accept_immigrant(FitStruct *fs);

    TSimulation *sim;

    size_t id;
    size_t nagents;
    std::default_random_engine rng;

    long generation_number = -1;
    std::vector<agent *> generation_agents;
    FittestList elites;
    FittestList prev_generation;
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

    int ndemes;
    long agents_per_deme;
    std::vector<Deme *> demes;
    std::vector<agent *> generation_agents;
    FittestList elites;
    FittestList prev_generation;
    int genome_len;

    long test_number = -1;
    long trial_number = -1;
    long trial_timestep = -1;
    long trial_end_sim_step = -1;    
    long generation_number = -1;

    vector<agent *> create_generation();

    void new_test();
    void new_trial();
    bool new_generation();
    void end_generation();

    void log_elite(FitStruct *fs);
};

extern TrialsState *trials;
#endif
