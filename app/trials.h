#pragma once

#define TRIALS true

#if TRIALS
#include <map>
#include <vector>

#define TEST_INTERLUDE 10
#define NTRIALS 6

class agent;
class TSimulation;

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

    virtual void end_generation(std::vector<long> &ranking) = 0;
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
    std::vector<agent *> get_agents();

    void init_test();
    void end_test();
    
    void init_trial();
    void end_trial();

    void end_generation();
};

extern TrialsState *trials;
#endif
