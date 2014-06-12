#include "datalib.h"
#include "PathDistance.h"
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

int get_training_trials_per_patch() {
    ifstream in("training_trials_per_patch.txt");
    int n;
    in >> n;
    db("Training trials per patch: " << n);
    return n;
}

float crow_dist(float x1, float z1, float x2, float z2) {
    float dx = x1-x2;
    float dz = z1-z2;

    return sqrt(dx*dx + dz*dz);
}

#if true
#define NTRIAL_EVALUATION 10
#else
#define NTRIAL_EVALUATION 2
#endif

TrialsState::TrialsState(TSimulation *sim_) {
    sim = sim_;
    trial_number = -1;
    curr_trial = nullptr;
    f = nullptr;
    
    int nfoodpatches = get_foodpatches_count();
    int ntrials_per_patch = get_training_trials_per_patch();

    ntrials_training = nfoodpatches * ntrials_per_patch;
    ntrials_evaluation = NTRIAL_EVALUATION;
    ntrials_total = ntrials_training + ntrials_evaluation;

    trials = new TrialState[ntrials_total];
    trials_evaluation = trials + ntrials_training;

    {
        for(int i = 0; i < nfoodpatches; i++) {
            for(int j = 0; j < ntrials_per_patch; j++) {
                food_sequence.push_back(i);
            }
        }
    }
    {
        assert(nfoodpatches == 2);
        int n = 0;
        for(int i = 0; (i < 2) && (n < ntrials_evaluation); i++)
            for(int j = 0; (j < 5) && (n < ntrials_evaluation); j++)
                food_sequence.push_back(i);
    }

    shuffle(food_sequence);

    sim->fMaxSteps = (ntrials_total * TRIAL_DURATION) + 1;
}

TrialsState::~TrialsState() {
    delete [] trials;
}

int TrialsState::get_foodpatches_count() {
    assert(sim->fNumDomains == 1);
    const int dom_index = 0;
    Domain &dom = sim->fDomains[dom_index];
    return dom.numFoodPatches;
}

void TrialsState::step() {
    if((curr_trial == nullptr) || (curr_trial->step == TRIAL_DURATION)) {
        init_trial();
    }
    curr_trial->step++;
}

void TrialsState::agent_success(agent *a) {
    db("  success @ " << curr_trial->step << ": " << a->Number());

    Fitness &fit = curr_trial->fitness[a->Number()];
    fit.success = true;
    fit.step_end = curr_trial->step;
    
    objectxsortedlist::gXSortedObjects.toMark( AGENTTYPE ); // point list back to c
	objectxsortedlist::gXSortedObjects.removeObjectWithLink( (gobject*) a );
    sim->fStage.RemoveObject(a);
    successful_agents.push_back(a);
}

void TrialsState::end_trial() {
    db("--- END TRIAL ---");

    vector<agent *> agents = get_agents();

    float origin_x = globals::worldsize/2.0f;
    float origin_z = -globals::worldsize/2.0f;

    float initial_dist = PathDistance::distance(origin_x, origin_z, f->x(), f->z());
    int food_segment = PathDistance::getSegmentId(f->x(), f->z());

    for(agent *a: agents) {
        Fitness &fit = curr_trial->fitness[a->Number()];

        // Compute final distance from food.
        {
            if(fit.success) {
                fit.final_dist_from_food = 0.0f;
                fit.dist_score = 1.0f;
            } else {
                fit.final_dist_from_food = PathDistance::distance(a->x(), a->z(), f->x(), f->z());
                fit.final_dist_from_origin = crow_dist(a->x(), a->z(), origin_x, origin_z);
                fit.dist_score = (MAX_DIST - fit.final_dist_from_food) / MAX_DIST;
            }
        }

        // Determine if on same distance path segment as food
        {
            fit.on_food_segment =
                fit.success ||
                (food_segment == PathDistance::getSegmentId(a->x(), a->z())
                 && fit.final_dist_from_origin > ORIGIN_TO_BRANCH_DIST);
        }

        // Compute score for trial
        {

            if(fit.success) {
                long time;
                if(agent::unfreezeStep > curr_trial->sim_start_step) {
                    time = (curr_trial->sim_start_step + fit.step_end) - agent::unfreezeStep;
                } else {
                    time = fit.step_end;
                }
                fit.velocity = initial_dist / time;
                fit.score = (0.5f * fit.dist_score) + (0.5f * fit.velocity) + 0.1f;
            } else if(fit.final_dist_from_food > initial_dist) {
                fit.score = 0.1f * fit.dist_score;
            } else if(fit.final_dist_from_origin < ORIGIN_TO_BRANCH_DIST) {
                fit.score = 0.1f * fit.dist_score;
            } else {
                fit.score = 0.5f * fit.dist_score;
            }
        }
    }
}

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

void TrialsState::init_trial() {
    trial_number++;

    if(trial_number > 0) {
        end_trial();
    }

    db("=== Starting trial " << trial_number << " (" << (trial_number>=ntrials_training?"TEST":"TRAIN") << ") ===");

    if(trial_number == ntrials_total) {
        end_trials();
    } else {
        curr_trial = &trials[trial_number];
        curr_trial->sim_start_step = sim->getStep();

        // Maintain agents
        if(trial_number > 0) {
            for(agent *a: successful_agents) {
                objectxsortedlist::gXSortedObjects.add(a);
                sim->fStage.AddObject(a);
            }
            successful_agents.clear();

            objectxsortedlist::gXSortedObjects.reset();
            agent *a;
            while (objectxsortedlist::gXSortedObjects.nextObj(AGENTTYPE, (gobject**)&a)) {
                a->setx(globals::worldsize/2.0f);
                a->setz(-globals::worldsize/2.0f);
                a->SaveLastPosition();
                a->setyaw(0.0);
                a->SetEnergy(a->GetMaxEnergy());
            }
            objectxsortedlist::gXSortedObjects.sort();
            objectxsortedlist::gXSortedObjects.reset();
        }

        // Maintain food
        {
            if(f) {
                food *_f;
                objectxsortedlist::gXSortedObjects.reset();
                objectxsortedlist::gXSortedObjects.nextObj(FOODTYPE, (gobject**)&_f);
                assert(f == _f);
                sim->RemoveFood(f);
            }

            assert(sim->fNumDomains == 1);
            const int dom_index = 0;
            int food_index = food_sequence[trial_number];
            f = sim->AddFood(dom_index, food_index);

            if(sim->fNumSoundPatches > 0) {
                sim->fSoundPatches[food_index].activate(sim->getStep() + 1);
            }
        }
    }
}

vector<agent *> TrialsState::get_agents() {
    vector<agent *> agents;

    for(agent *a: successful_agents) {
        agents.push_back(a);
    }
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