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

bool is_voice_activated(agent *a) {
    return a->Voice() > 0.01f;
}

template<typename Tfit>
struct TestImpl : public Test
{
    std::map<long, Tfit> fitness[NTRIALS];

    virtual ~TestImpl() {}
    Tfit &get(int trial_number, agent *a) { return get(trial_number, a->Number()); }
    Tfit &get(int trial_number, long agent_number) { return fitness[trial_number][agent_number]; }
};

struct Step0Fitness {
    long nactivated = 0;
};

struct Step0 : public TestImpl<Step0Fitness>
{
    virtual ~Step0() {}

    virtual long get_step_count() {
        return 50;
    }

    virtual void evaluate_step(int trial_number, long test_step, agent *a, int freq) {
        auto &fitness = get(trial_number, a);

        make_sound(a, freq);
        if(is_voice_activated(a)) {
            fitness.nactivated++;
        }
    }

    virtual float get_trial_score(int trial_number, agent *a) {
        auto fitness = get(trial_number, a);
        return float(fitness.nactivated) / get_step_count();
    }

    virtual float get_test_score(vector<float> &trial_scores) {
        return mean(trial_scores);
    }

    virtual void end_generation(vector<long> ranking) {
        // Trials
        {
            static const char *colnames[] = {
                "Trial", "ActivatedCount", "Score", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     INT,              FLOAT
            };

            DataLibWriter *writer = new DataLibWriter( "run/step0-trials.log", true, true );

            for(long agent_number: ranking) {
                char tableName[32];
                sprintf(tableName, "Agent%ld", agent_number);

                writer->beginTable( tableName,
                                    colnames,
                                    coltypes );

                auto &trial_scores = this->trial_scores[agent_number];

                for(int i = 0; i < NTRIALS; i++) {
                    auto fit = get(i, agent_number);

                    writer->addRow( i, fit.nactivated, trial_scores[i] );

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

            DataLibWriter *writer = new DataLibWriter( "run/step0-test.log", true, true );

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
        long test_step = trial_step - TEST_INTERLUDE;
        //db("  --- test step " << test_step << " @ " << sim->getStep());
        int freq = freq_sequence[trial_number];
        auto test = tests[test_number];
        for(agent *a: get_agents()) {
            test->evaluate_step(trial_number, test_step, a, freq);
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
