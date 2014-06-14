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

    virtual long get_step_count() = 0;
    virtual void evaluate_step(int trial_number, long test_step, agent *a, int freq) = 0;
    virtual float get_trial_score(int trial_number, agent *a) = 0;
    virtual float get_test_score(std::vector<float> &trial_scores) = 0;

    virtual void end_generation(std::vector<long> ranking) = 0;

    std::map<long, std::vector<float>> trial_scores;
    std::map<long, float> test_scores;
};

struct TrialsState
{
    TrialsState(TSimulation *sim_);
    ~TrialsState();

    void step();

    TSimulation *sim;
    std::vector<int> freq_sequence;

    std::vector<Test *> tests;

    long test_number;
    long trial_number;
    long trial_step;
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
