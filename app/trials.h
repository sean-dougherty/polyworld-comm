#pragma once

#define TRIALS true

#if TRIALS
#include <map>
#include <vector>

#define TRIAL_DURATION 180

#define MAX_DIST (190.0f/2.0f)

class agent;
class TSimulation;

struct Fitness
{
    bool success = false;
    long step_end = -1;
    float final_dist_from_food = -1.0f;
    float final_dist_from_origin = 0.0f;
    float velocity = 0.0f;
    float dist_score = 0.0f;
    float score = 0.0f;
};

struct TotalFitness
{
    agent *a;
    int nsuccesses = 0;
    float score = 0.0f;
    float trial_score_mean = 0.0f;
    float trial_score_stddev = 0.0f;
};

struct TrialState
{
    long step = 0;
    long sim_start_step = 0;

    std::map<long, Fitness> fitness;
};

struct TrialsState
{
    TrialsState(TSimulation *sim_);
    ~TrialsState();

    void step();
    void agent_success(class agent *a);

    TSimulation *sim;
    long trial_number;
    TrialState *trials;
    TrialState *trials_evaluation;
    std::vector<int> food_sequence;
    
    TrialState *curr_trial;
    
    int ntrials_training;
    int ntrials_evaluation;
    int ntrials_total;
    
private:
    std::vector<agent *> get_agents();
    int get_foodpatches_count();

    std::vector<agent *> successful_agents;

    void init_trial();
    void end_trial();
    void end_trials();
    class food *f;
};

extern TrialsState *trials;
#endif
