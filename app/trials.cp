#include "datalib.h"
#include "PathDistance.h"
#include "Retina.h"
#include "trials.h"
#include "Simulation.h"
#include "SoundPatch.h"

#include <algorithm>

using namespace datalib;
using namespace std;

#if TRIALS

#define DEBUG true

#if DEBUG
#define db(x...) cout << x << endl;
#else
#define db(x...)
#endif

TrialsState *trials = nullptr;

float mean(vector<float> scores);
float stddev(vector<float> scores);

int max_repeat(vector<int> &x) {
    int maxrepeat = 0;
    int nrepeat = 0;
    int val = -1;

    for(int y: x) {
        if(y != val) {
            maxrepeat = max(maxrepeat, nrepeat);
            val = y;
            nrepeat = 1;
        } else {
            nrepeat++;
        }
    }

    return maxrepeat;
 }

void shuffle(vector<int> &x) {
    ifstream in("trial_seed.txt");
    int seed;
    in >> seed;
    db("Food sequence RNG seed: " << seed);

    auto rng = std::default_random_engine(seed);

    do {
        shuffle(x.begin(), x.end(), rng);
    } while(max_repeat(x) > 4);
}

void show_color(agent *a, float r, float g, float b) {
    a->GetRetina()->force_color(r, g, b);
}

void show_green(agent *a) {
    show_color(a, 0.0, 1.0, 0.0);
}

void show_black(agent *a) {
    show_color(a, 0.0, 0.0, 0.0);
}

void make_sound(agent *a, int freq) {
    a->sound(1.0, freq, a->x(), a->z());
}

#define make_silence(a) // no-op

float get_voice_activation(agent *a) {
    return a->Voice();
}

int get_voice_frequency(agent *a) {
    assert(false);
    return 0;
}

#define count_score(COUNT_NAME) (float(trial_state.COUNT_NAME##_count) / get_trial_timestep_count())

template<typename Tfit>
struct TestImpl : public Test
{
    std::map<long, Tfit> fitness[NTRIALS];

    virtual ~TestImpl() {}
    Tfit &get(int trial_number, agent *a) { return get(trial_number, a->Number()); }
    Tfit &get(int trial_number, long agent_number) { return fitness[trial_number][agent_number]; }
};

struct Test0TrialState {
    vector<float> x;
    vector<float> y;
    float covariance;
    float score;
};

struct Test0 : public TestImpl<Test0TrialState>
{
    const long Timesteps_on = 10;
    const long Timesteps_off = 5;

    const long Phase0_end = Timesteps_on;
    const long Phase1_end = Phase0_end + Timesteps_off;

    virtual ~Test0() {}

    virtual long get_trial_timestep_count() {
        return Phase1_end;
    }

    virtual void timestep_input(int trial_number,
                                long time,
                                agent *a,
                                int freq) {

        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            show_black(a);
            make_sound(a, freq);
            trial_state.x.push_back(1.0f);
        } else {
            show_black(a);
            make_silence(a);
            trial_state.x.push_back(0.0f);
        }
    }

    virtual void timestep_output(int trial_number,
                                 long time,
                                 agent *a,
                                 int freq) {

        auto &trial_state = get(trial_number, a);

        trial_state.y.push_back(get_voice_activation(a));
    }

    virtual float get_trial_score(int trial_number, agent *a) {
        auto trial_state = get(trial_number, a);

        trial_state.covariance = covariance(trial_state.x, trial_state.y);
        trial_state.score = -1.0f;
        assert(false); // make score!
        return trial_state.score;
    }

    virtual float get_test_score(vector<float> &trial_scores) {
        return mean(trial_scores);
    }

    virtual void end_generation(vector<long> ranking) {
        // Trials
        {
            static const char *colnames[] = {
                "Trial", "Covariance", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT,        FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test0-trials.log", true, true );

            for(long agent_number: ranking) {
                char tableName[32];
                sprintf(tableName, "Agent%ld", agent_number);

                writer->beginTable( tableName,
                                    colnames,
                                    coltypes );

                for(int i = 0; i < NTRIALS; i++) {
                    auto trial_state = get(i, agent_number);

                    writer->addRow( i,
                                    trial_state.covariance,
                                    trial_state.score );

                }

                writer->endTable();
            }

            delete writer;
        }

        // Test
        {
            static const char *colnames[] = {
                "Agent", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test0-test.log", true, true );

            writer->beginTable( "Scores",
                                colnames,
                                coltypes );

            for(long agent_number: ranking) {
                writer->addRow( agent_number,
                                test_scores[agent_number] );
            }

            writer->endTable();

            delete writer;
        }
    }
};


struct Test1TrialState {
    int correspondence_count = 0;
    int respond_count = 0;
    int break_count = 0;
    float score;
};

struct Test1 : public TestImpl<Test1TrialState>
{
    const long Timesteps_on = 10;
    const long Timesteps_off = 5;

    const long Phase0_end = Timesteps_on;
    const long Phase1_end = Phase0_end + Timesteps_off;

    virtual ~Test1() {}

    virtual long get_trial_timestep_count() {
        return Phase1_end;
    }

    virtual void timestep_input(int trial_number,
                                long time,
                                agent *a,
                                int freq) {
        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            show_green(a);
            make_sound(a, freq);
        } else {
            show_black(a);
            make_silence(a);
        }
    }

    virtual void timestep_output(int trial_number,
                                 long time,
                                 agent *a,
                                 int freq) {
        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            if(is_voicing(a)) {
                trial_state.respond_count++;
            }

            if(freq == get_voice_frequency(a)) {
                trial_state.correspondence_count++;
            }
        } else {
            if(!is_voicing(a)) {
                trial_state.break_count++;
            }
        }
    }

    virtual float get_trial_score(int trial_number, agent *a) {
        auto trial_state = get(trial_number, a);

        trial_state.score =
            (0.333f * count_score(correspondence))
            + (0.333f * count_score(respond))
            + (0.333f * count_score(break));

        return trial_state.score;
    }

    virtual float get_test_score(vector<float> &trial_scores) {
        return mean(trial_scores);
    }

    virtual void end_generation(vector<long> ranking) {
        // Trials
        {
            static const char *colnames[] = {
                "Trial", "Correspondence", "Respond", "Break", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT,            FLOAT,     FLOAT,   FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test1-trials.log", true, true );

            for(long agent_number: ranking) {
                char tableName[32];
                sprintf(tableName, "Agent%ld", agent_number);

                writer->beginTable( tableName,
                                    colnames,
                                    coltypes );

                for(int i = 0; i < NTRIALS; i++) {
                    auto trial_state = get(i, agent_number);

                    writer->addRow( i,
                                    count_score(correspondence),
                                    count_score(respond),
                                    count_score(break),
                                    trial_state.score );
                }

                writer->endTable();
            }

            delete writer;
        }

        // Test
        {
            static const char *colnames[] = {
                "Agent", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test1-test.log", true, true );

            writer->beginTable( "Scores",
                                colnames,
                                coltypes );

            for(long agent_number: ranking) {
                writer->addRow( agent_number,
                                test_scores[agent_number] );
            }

            writer->endTable();

            delete writer;
        }
    }
};

struct Test2TrialState {
    int correspondence_count = 0;
    int respond_count = 0;
    int delay_count = 0;
    float score;
};

struct Test2 : public TestImpl<Test2TrialState>
{
    const long Timesteps_green_off = 10;
    const long Timesteps_green_on = 10;

    const long Phase0_end = Timesteps_green_off;
    const long Phase1_end = Phase0_end + Timesteps_green_on;

    virtual ~Test2() {}

    virtual long get_trial_timestep_count() {
        return Phase1_end;
    }

    virtual void timestep_input(int trial_number,
                                long time,
                                agent *a,
                                int freq) {
        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            show_black(a);
            make_sound(a, freq);
        } else {
            show_green(a);
            make_sound(a, freq);
        }
    }

    virtual void timestep_output(int trial_number,
                                 long time,
                                 agent *a,
                                 int freq) {
        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            if(!is_voicing(a)) {
                trial_state.delay_count++;
            }
        } else {
            if(is_voicing(a)) {
                trial_state.respond_count++;
            }

            if(freq == get_voice_frequency(a)) {
                trial_state.correspondence_count++;
            }
        }
    }

    virtual float get_trial_score(int trial_number, agent *a) {
        auto trial_state = get(trial_number, a);

        trial_state.score =
            (0.333f * count_score(correspondence))
            + (0.333f * count_score(respond))
            + (0.333f * count_score(delay));

        return trial_state.score;
    }

    virtual float get_test_score(vector<float> &trial_scores) {
        return mean(trial_scores);
    }

    virtual void end_generation(vector<long> ranking) {
        // Trials
        {
            static const char *colnames[] = {
                "Trial", "Correspondence", "Respond", "Delay", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT,            FLOAT,      FLOAT,  FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test2-trials.log", true, true );

            for(long agent_number: ranking) {
                char tableName[32];
                sprintf(tableName, "Agent%ld", agent_number);

                writer->beginTable( tableName,
                                    colnames,
                                    coltypes );

                for(int i = 0; i < NTRIALS; i++) {
                    auto trial_state = get(i, agent_number);

                    writer->addRow( i,
                                    count_score(correspondence),
                                    count_score(respond),
                                    count_score(delay),
                                    trial_state.score );
                }

                writer->endTable();
            }

            delete writer;
        }

        // Test
        {
            static const char *colnames[] = {
                "Agent", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test2-test.log", true, true );

            writer->beginTable( "Scores",
                                colnames,
                                coltypes );

            for(long agent_number: ranking) {
                writer->addRow( agent_number, test_scores[agent_number] );
            }

            writer->endTable();

            delete writer;
        }
    }
};

struct Test4TrialState {
    int correspondence_count = 0;
    int respond_count = 0;
    int delay_count = 0;
    int break_count = 0;
    float score;
};

struct Test4 : public TestImpl<Test4TrialState>
{
    const long Timesteps_green_off0 = 10;
    const long Timesteps_green_on = 10;
    const long Timesteps_green_off1 = 5;

    const long Phase0_end = Timesteps_green_off0;
    const long Phase1_end = Phase0_end + Timesteps_green_on;
    const long Phase2_end = Phase1_end + Timesteps_green_off1;

    virtual ~Test4() {}

    virtual long get_trial_timestep_count() {
        return Phase2_end;
    }

    virtual void timestep_input(int trial_number,
                                long time,
                                agent *a,
                                int freq) {

        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            show_black(a);
            make_sound(a, freq);
        } else if(time <= Phase1_end) {
            show_green(a);
            make_silence(a);
        } else {
            show_black(a);
            make_silence(a);
        }
    }

    virtual void timestep_output(int trial_number,
                                 long time,
                                 agent *a,
                                 int freq) {
        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            if(!is_voicing(a)) {
                trial_state.delay_count++;
            }
        } else if(time <= Phase1_end) {
            if(is_voicing(a)) {
                trial_state.respond_count++;
            }
            if(freq == get_voice_frequency(a)) {
                trial_state.correspondence_count++;
            }
        } else {
            if(!is_voicing(a)) {
                trial_state.break_count++;
            }
        }
    }

    virtual float get_trial_score(int trial_number, agent *a) {
        auto trial_state = get(trial_number, a);

        trial_state.score =
            (0.25f * count_score(correspondence))
            + (0.25f * count_score(respond))
            + (0.25f * count_score(delay))
            + (0.25f * count_score(break));

        return trial_state.score;
    }

    virtual float get_test_score(vector<float> &trial_scores) {
        return mean(trial_scores);
    }

    virtual void end_generation(vector<long> ranking) {
/*
        // Trials
        {
            static const char *colnames[] = {
                "Trial", "Correspondence", "Respond", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     INT,               INT,      FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test1-trials.log", true, true );

            for(long agent_number: ranking) {
                char tableName[32];
                sprintf(tableName, "Agent%ld", agent_number);

                writer->beginTable( tableName,
                                    colnames,
                                    coltypes );

                for(int i = 0; i < NTRIALS; i++) {
                    auto trial_state = get(i, agent_number);

                    writer->addRow( i, trial_state.correspondence_count, trial_state.respond_count, trial_state.score );

                }

                writer->endTable();
            }

            delete writer;
        }

        // Test
        {
            static const char *colnames[] = {
                "Agent", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/step1-test.log", true, true );

            writer->beginTable( "Scores",
                                colnames,
                                coltypes );

            for(long agent_number: ranking) {
                writer->addRow( agent_number, test_scores[agent_number] );
            }

            writer->endTable();

            delete writer;
        }
*/
    }
};

struct Test5TrialState {
    int correspondence_count = 0;
    int respond_count = 0;
    int delay_count = 0;
    int break_count = 0;
    float score;
};

struct Test5 : public TestImpl<Test5TrialState>
{
    const long Timesteps_sound_on = 10;
    const long Timesteps_sound_off = 5;
    const long Timesteps_green_on = 10;
    const long Timesteps_green_off = 5;

    const long Phase0_end = Timesteps_sound_on;
    const long Phase1_end = Phase0_end + Timesteps_sound_off;
    const long Phase2_end = Phase1_end + Timesteps_green_on;
    const long Phase3_end = Phase2_end + Timesteps_green_off;

    virtual ~Test5() {}

    virtual long get_trial_timestep_count() {
        return Timesteps_sound_on
            + Timesteps_sound_off
            + Timesteps_green_on
            + Timesteps_green_off;
    }

    virtual void timestep_input(int trial_number,
                                long time,
                                agent *a,
                                int freq) {

        auto &trial_state = get(trial_number, a);

        if(time <= Phase0_end) {
            show_black(a);
            make_sound(a, freq);
        } else if(time <= Phase1_end) {
            show_black(a);
            make_silence(a);
        } else if(time <= Phase2_end) {
            show_green(a);
            make_silence(a);
        } else {
            show_black(a);
            make_silence(a);
        }
    }

    virtual void timestep_output(int trial_number,
                                 long time,
                                 agent *a,
                                 int freq) {

        auto &trial_state = get(trial_number, a);

        if(time <= Phase1_end) {
            if(!is_voicing(a)) {
                trial_state.delay_count++;
            }
        } else if(time <= Phase2_end) {
            if(is_voicing(a)) {
                trial_state.respond_count++;
            }
            if(freq == get_voice_frequency(a)) {
                trial_state.correspondence_count++;
            }
        } else {
            if(!is_voicing(a)) {
                trial_state.break_count++;
            }
        }
    }

    virtual float get_trial_score(int trial_number, agent *a) {
        auto trial_state = get(trial_number, a);

        trial_state.score =
            (0.25f * count_score(correspondence)
             + (0.25f * count_score(respond))
             + (0.25f * count_score(delay_score))
             + (0.25f * count_score(break_score));

        return trial_state.score;
    }

    virtual float get_test_score(vector<float> &trial_scores) {
        return mean(trial_scores);
    }

    virtual void end_generation(vector<long> ranking) {
/*
        // Trials
        {
            static const char *colnames[] = {
                "Trial", "Correspondence", "Respond", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     INT,               INT,      FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/test1-trials.log", true, true );

            for(long agent_number: ranking) {
                char tableName[32];
                sprintf(tableName, "Agent%ld", agent_number);

                writer->beginTable( tableName,
                                    colnames,
                                    coltypes );

                for(int i = 0; i < NTRIALS; i++) {
                    auto trial_state = get(i, agent_number);

                    writer->addRow( i, trial_state.correspondence_count, trial_state.respond_count, trial_state.score );

                }

                writer->endTable();
            }

            delete writer;
        }

        // Test
        {
            static const char *colnames[] = {
                "Agent", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/step1-test.log", true, true );

            writer->beginTable( "Scores",
                                colnames,
                                coltypes );

            for(long agent_number: ranking) {
                writer->addRow( agent_number, test_scores[agent_number] );
            }

            writer->endTable();

            delete writer;
        }
*/
    }
};

TrialsState::TrialsState(TSimulation *sim_) {
    sim = sim_;
    test_number = -1;
    trial_number = -1;

    tests.push_back(new Step0());
    
    long nsteps = 0;
    for(auto test: tests) {
        nsteps += NTRIALS * (test->get_step_count() + TEST_INTERLUDE);
    }

    sim->fMaxSteps = nsteps + 1;

    for(int freq = 0; freq < 2; freq++) {
        for(int i = 0; i < NTRIALS/2; i++) {
            freq_sequence.push_back(i);
        }
    }

    shuffle(freq_sequence);
}

TrialsState::~TrialsState() {
}

void TrialsState::step() {
    if(test_number == -1) {
        test_number = 0;
        trial_number = 0;
        init_test();
        init_trial();
    } else if(sim->getStep() == trial_end_sim_step) {
        end_trial();
        trial_number++;
        if(trial_number == NTRIALS) {
            end_test();
            test_number++;
            if(test_number == (int)tests.size()) {
                end_generation();
                return;
            } else {
                trial_number = 0;
                init_test();
                init_trial();
            }
        } else {
            init_trial();
        }
    }

    trial_step++;

    if(trial_step > TEST_INTERLUDE) {
        long test_timestep = trial_step - TEST_INTERLUDE;
        //db("  --- test step " << test_timestep << " @ " << sim->getStep());
        int freq = freq_sequence[trial_number];
        auto test = tests[test_number];
        for(agent *a: get_agents()) {
            test->evaluate_step(trial_number, test_timestep, a, freq);
        }
    }
}

void TrialsState::init_test() {
    db("=== Beginning test " << test_number);
}

void TrialsState::end_test() {
    db("=== Ending test " << test_number);

    auto test = tests[test_number];
    for(agent *a: get_agents()) {
        float test_score = test->get_test_score(test->trial_scores[a->Number()]);
        test->test_scores[a->Number()] = test_score;
    }
}

void TrialsState::init_trial() {
    db("*** Beginning trial " << trial_number << " of test " << test_number);

    auto test = tests[test_number];
    trial_step = 0;
    trial_end_sim_step = sim->getStep() + TEST_INTERLUDE + test->get_step_count();

    for(agent *a: get_agents()) {
        a->SetEnergy(a->GetMaxEnergy());
        a->setx(0.0);
        a->setz(0.0);
        show_black(a);
    }
}

void TrialsState::end_trial() {
    db("*** Ending trial " << trial_number << " of test " << test_number);

    auto test = tests[test_number];
    for(agent *a: get_agents()) {
        float score = test->get_trial_score(trial_number, a);
        test->trial_scores[a->Number()].push_back(score);
    }
}

struct TotalFitness {
    long agent_number = 0;
    vector<float> test_scores;
    float score = 0;
};

void TrialsState::end_generation() {
    db("END OF GENERATION");
    sim->End("endTests");

    map<long, TotalFitness> total_fitnesses_lookup;

    for(auto test: tests) {
        for(auto test_score: test->test_scores) {
            long agent_number = test_score.first;
            float score = test_score.second;

            TotalFitness &fitness = total_fitnesses_lookup[agent_number];
            fitness.agent_number = agent_number;
            fitness.test_scores.push_back(score);
            fitness.score += score;
        }
    }

    vector<TotalFitness> total_fitnesses;
    for(auto fitness: total_fitnesses_lookup) {
        total_fitnesses.push_back(fitness.second);
    }

    for(auto fitness: total_fitnesses) {
        fitness.score = mean(fitness.test_scores);
    }

    sort(total_fitnesses.begin(), total_fitnesses.end(),
         [](const TotalFitness &x, const TotalFitness &y) {
             return y.score < x.score;
         });

    vector<long> ranking;

    for(auto fitness: total_fitnesses) {
        ranking.push_back(fitness.agent_number);
    }

    for(auto test: tests) {
        test->end_generation(ranking);
    }

    map<long, agent *> agents;
    for(agent *a: get_agents()) {
        agents[a->Number()] = a;
    }

    system("mkdir -p run/genome/Fittest");
    {
        FILE *ffitness = fopen( "run/genome/Fittest/fitness.txt", "w" );

        for(int i = 0; i < 10; i++)
        {
            TotalFitness &fit = total_fitnesses[i];
            fprintf( ffitness, "%ld %f\n", fit.agent_number, fit.score );

            {
                genome::Genome *g = agents[fit.agent_number]->Genes();
                char path[256];
                sprintf( path, "run/genome/Fittest/genome_%ld.txt", fit.agent_number );
                AbstractFile *out = AbstractFile::open(globals::recordFileType, path, "w");
                g->dump(out);
                delete out;
            }

        }

        fclose( ffitness );
    }

}

#if false
void TrialsState::end_trials() {
    vector<TotalFitness> total_fits;

    db("END TRIALS");

    for(agent *a: get_agents()) {
        TotalFitness total_fit;
        total_fit.a = a;

        vector<float> agent_scores;
        vector<float> velocity;
        for(int i = 0; i < ntrials_evaluation; i++) {
            Fitness &fit = trials_evaluation[i].fitness[a->Number()];

            if(fit.success) {
                total_fit.nsuccesses++;
            }
            if(fit.on_food_segment) {
                total_fit.non_food_segment++;
            }
            velocity.push_back(fit.velocity);
            agent_scores.push_back(fit.score);
        }

        total_fit.velocity_mean = mean(velocity);
        total_fit.velocity_stddev = stddev(velocity);
        total_fit.trial_score_mean = mean(agent_scores);
        total_fit.trial_score_stddev = stddev(agent_scores);

        float mean_score = total_fit.trial_score_mean;
        float stddev_score = 1.0f - total_fit.trial_score_stddev;
        total_fit.score = (0.95f * mean_score) + (0.05f * stddev_score);

        total_fits.push_back(total_fit);
    }


    sort( total_fits.begin(), total_fits.end(),
          [](const TotalFitness &x, const TotalFitness &y) {
              return y.score < x.score;
          } );

    // Create trials log file
    {
        DataLibWriter *writer = new DataLibWriter( "run/trials.log", true, true );

        for(auto total_fit: total_fits) {
            agent *a = total_fit.a;

            char tableName[32];
            sprintf(tableName, "Agent%ld", a->Number());

            static const char *colnames[] = {
                "Trial", "Success", "OnFoodSegment", "FoodDist", "OriginDist", "Velocity", "DistScore", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     BOOL,      BOOL,            FLOAT,      FLOAT,        FLOAT,      FLOAT,       FLOAT
            };

            writer->beginTable( tableName,
                                colnames,
                                coltypes );

            for(int i = 0; i < ntrials_evaluation; i++) {
                Fitness &fit = trials_evaluation[i].fitness[a->Number()];

                writer->addRow( i, fit.success, fit.on_food_segment, fit.final_dist_from_food, fit.final_dist_from_origin, fit.velocity, fit.dist_score, fit.score );

            }

            writer->endTable();
        }

        delete writer;
    }

    // Create fitness log
    {
        DataLibWriter *writer = new DataLibWriter( "run/fitness.log", true, true );

        for(auto total_fit: total_fits) {
            agent *a = total_fit.a;

            char tableName[32];
            sprintf(tableName, "Agent%ld", a->Number());

            static const char *colnames[] = {
                "SuccessCount", "OnFoodSegmentCount", "VelocityMean", "VelocityStdDev", "TrialScoreMean", "TrialScoreStdDev", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,            INT,                  FLOAT,          FLOAT,            FLOAT,            FLOAT,              FLOAT
            };

            writer->beginTable( tableName,
                                colnames,
                                coltypes );

            writer->addRow( total_fit.nsuccesses, total_fit.non_food_segment, total_fit.velocity_mean, total_fit.velocity_stddev, total_fit.trial_score_mean, total_fit.trial_score_stddev, total_fit.score );


            writer->endTable();
        }

        delete writer;
    }

    system("mkdir -p run/genome/Fittest");
    {
        FILE *ffitness = fopen( "run/genome/Fittest/fitness.txt", "w" );

        for(int i = 0; i < 10; i++)
        {
            TotalFitness &fit = total_fits[i];
            fprintf( ffitness, "%ld %f\n", fit.a->Number(), fit.score );

            {
                genome::Genome *g = fit.a->Genes();
                char path[256];
                sprintf( path, "run/genome/Fittest/genome_%ld.txt", fit.a->Number() );
                AbstractFile *out = AbstractFile::open(globals::recordFileType, path, "w");
                g->dump(out);
                delete out;
            }

        }

        fclose( ffitness );
    }

    // Deal with most successful agent
    {
        TotalFitness &total_fit = total_fits.front();
        bool success = true;

        // Success/fail
        {
            if( (float(total_fit.nsuccesses) / ntrials_evaluation) < 0.9f ) {
                db("not enough successful trials");
                success = false;
            } else if( total_fit.trial_score_mean < 0.75 ) {
                db("trials score mean too low");
                success = false;
            } else if( total_fit.trial_score_stddev > 0.1 ) {
                db("trials score stddev too high");
                success = false;
            }

            FILE *f = fopen("run/trials_result.txt", "w");
            fprintf(f, "%s\n", success ? "success" : "fail");
            fclose(f);
        }

        // Velocity
        {
            FILE *f = fopen("run/velocity.txt", "w");
            fprintf(f, "%f %f\n", total_fit.velocity_mean, total_fit.velocity_stddev);
            fclose(f);
        }

        // FoodSegment
        {
            FILE *f = fopen("run/onfoodsegment.txt", "w");
            fprintf(f, "%f\n", float(total_fit.non_food_segment) / ntrials_evaluation);
            fclose(f);
        }
    }

    sim->End("trialsComplete");
}
#endif

vector<agent *> TrialsState::get_agents() {
    vector<agent *> agents;

    {
        agent *a;
        objectxsortedlist::gXSortedObjects.reset();
        while (objectxsortedlist::gXSortedObjects.nextObj(AGENTTYPE, (gobject**)&a)) {
            agents.push_back(a);
        }
    }

    return agents;
}

float mean(vector<float> scores) {
    float sum = 0.0f;
    for(auto x: scores)
        sum += x;
    return sum / scores.size();
}

float stddev(vector<float> scores) {
    const float N = scores.size();
    float sum = 0.0f;
    float sum2 = 0.0f;
    for(auto x: scores) {
        sum += x;
        sum2 += x*x;
    }
    float result = sqrt( (sum2 - (sum*sum) / N) / N );
    return result;
}

#endif
