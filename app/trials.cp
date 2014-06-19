#include "datalib.h"
#include "GenomeUtil.h"
#include "PathDistance.h"
#include "Retina.h"
#include "trials.h"
#include "Simulation.h"
#include "SoundPatch.h"

#include <algorithm>
#include <string>

using namespace datalib;
using namespace genome;
using namespace std;

#define GENERATION_LOG_FREQUENCY 100

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
float covariance(vector<float> &x, vector<float> &y);
void shuffle(vector<int> &x, int seed);

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
    return trials->sim->getVoiceFrequency(a);
}

bool is_voicing(agent *a) {
    return get_voice_frequency(a) >= 0;
}

enum TaskCategory {
    Delay,
    Speak,
    Break
};

enum Sound {
    Freq,
    Silent
};

enum class Metric : int {
    Delay = 0,
    Correspondence = 1,
    Respond = 2,
    Break = 3,
    __N = 4
};

struct Task {
    TaskCategory category;
    Sound sound;
    long timesteps;

    struct AgentTrialState {
        long silence_count = 0;
        long correspondence_count = 0;
        long respond_count = 0;
    };
    map<long, AgentTrialState> agent_trial_state[NTRIALS];

    void timestep_input(agent *a, int freq) {
        //
        // Sound
        //
        if(sound == Freq) {
            make_sound(a, freq);
        } else {
            make_silence(a);
        }

        //
        // Vision
        //
        switch(category) {
        case Delay:
        case Break:
            show_black(a);
            break;
        case Speak:
            show_green(a);
            break;
        default:
            assert(false);
        }
    }

    void timestep_output(int trial_number,
                         agent *a,
                         int freq) {
        AgentTrialState &state = agent_trial_state[trial_number][a->Number()];

        switch(category) {
        case Delay:
        case Break:
            if(!is_voicing(a)) {
                state.silence_count++;
            }
            break;
        case Speak:
            if(is_voicing(a)) {
                state.respond_count++;
            }
            if(freq == get_voice_frequency(a)) {
                state.correspondence_count++;
            }
            break;
        default:
            assert(false);
            break;
        }
    }

    float metric(int trial_number, long agent_number, Metric m) {
        AgentTrialState &state = agent_trial_state[trial_number][agent_number];

        switch(m) {
        case Metric::Delay:
            assert(category == Delay);
            return float(state.silence_count) / timesteps;
        case Metric::Respond:
            assert(category == Speak);
            return float(state.respond_count) / timesteps;
        case Metric::Correspondence:
            assert(category == Speak);
            return float(state.correspondence_count) / timesteps;
        case Metric::Break:
            assert(category == Break);
            return float(state.silence_count) / timesteps;
        default:
            assert(false);
        }
    }

    void create_log_schema(string prefix,
                           vector<string> &colnames,
                           vector<datalib::Type> &coltypes) {
        

        switch(category) {
        case Delay:
            colnames.push_back(prefix+"Delay");
            coltypes.push_back(datalib::FLOAT);
            break;
        case Speak:
            colnames.push_back(prefix+"Respond");
            coltypes.push_back(datalib::FLOAT);
            colnames.push_back(prefix+"Correspondence");
            coltypes.push_back(datalib::FLOAT);
            break;
        case Break:
            colnames.push_back(prefix+"Break");
            coltypes.push_back(datalib::FLOAT);
            break;
        default:
            assert(false);
        }
    }

    void log_trial(int trial_number, long agent_number, vector<Variant> &colvalues) {
        switch(category) {
        case Delay:
            colvalues.push_back(metric(trial_number, agent_number, Metric::Delay));
            break;
        case Speak:
            colvalues.push_back(metric(trial_number, agent_number, Metric::Respond));
            colvalues.push_back(metric(trial_number, agent_number, Metric::Correspondence));
            break;
        case Break:
            colvalues.push_back(metric(trial_number, agent_number, Metric::Break));
            break;
        default:
            assert(false);
        }
    }

    void reset() {
        for(int i = 0; i < NTRIALS; i++) {
            agent_trial_state[i].clear();
        }
    }
};

struct ScoredTest : public Test {
    const char *name;
    float weight;
    vector<Task> tasks;
    vector<long> task_ends;
    vector<int> tasks_by_metric[int(Metric::__N)];

ScoredTest(const char *name_,
           float weight_,
           vector<Task> tasks)
        : name(name_)
        , weight(weight_)
        , tasks(tasks)
    {
        long end = 0;

        for(auto &t: tasks) {
            end += t.timesteps;
            task_ends.push_back(end);
        }

        for(int i = 0; i < (int)tasks.size(); i++) {
            Task &t = tasks[i];
            switch(t.category) {
            case Delay:
                tasks_by_metric[int(Metric::Delay)].push_back(i);
                break;
            case Speak:
                tasks_by_metric[int(Metric::Correspondence)].push_back(i);
                tasks_by_metric[int(Metric::Respond)].push_back(i);
                break;
            case Break:
                tasks_by_metric[int(Metric::Break)].push_back(i);
                break;
            }
        }
    }

    virtual ~ScoredTest() {
    }

    virtual long get_trial_timestep_count() {
        return task_ends.back();
    }

    virtual void timestep_input(int trial_number,
                                long test_timestep,
                                agent *a,
                                int freq) {

        for(size_t i = 0; i < tasks.size(); i++) {
            if(test_timestep <= task_ends[i]) {
                tasks[i].timestep_input(a, freq);
                break;
            }
        }
    }

    virtual void timestep_output(int trial_number,
                                 long test_timestep,
                                 agent *a,
                                 int freq) {

        for(size_t i = 0; i < tasks.size(); i++) {
            if(test_timestep <= task_ends[i]) {
                tasks[i].timestep_output(trial_number, a, freq);
                break;
            }
        }
    }

    virtual void end_generation(long generation_number,
                                std::vector<long> &ranking) {
        if(generation_number % GENERATION_LOG_FREQUENCY == 0) {
            vector<string> colnames;
            vector<datalib::Type> coltypes;

            colnames.push_back("Trial");
            coltypes.push_back(datalib::INT);

            colnames.push_back("Freq");
            coltypes.push_back(datalib::INT);

            int prefix = 0;
            for(Task &t: tasks) {
                char prefix_str[32];
                sprintf(prefix_str, "%d.", prefix);
                t.create_log_schema(prefix_str, colnames, coltypes);
                prefix++;
            }

            {
                char path[512];
                sprintf(path, "run/generations/%ld/%s-trial-metrics.log", generation_number, name);        
                DataLibWriter *writer = new DataLibWriter( path, true, true );
                vector<Variant> colvalues;

                for(long agent_number: ranking) {
                    char tableName[32];
                    sprintf(tableName, "Agent%ld", agent_number);
                    writer->beginTable(tableName, colnames, coltypes);

                    for(int trial = 0; trial < NTRIALS; trial++) {
                        colvalues.clear();
                        colvalues.push_back(trial);
                        colvalues.push_back(trials->freq_sequence[trial]);

                        for(Task &t: tasks) {
                            t.log_trial(trial, agent_number, colvalues);
                        }

                        writer->addRow(&colvalues.front());
                    }

                    writer->endTable();
                }

                delete writer;
            }
        }
    }

    virtual void reset() {
        for(Task &t: tasks) {
            t.reset();
        }
    }

    float metric(agent *a, Metric m) {
        vector<int> &metric_tasks = tasks_by_metric[int(m)];

        float sum = 0.0f;

        for(int i: metric_tasks) {
            Task &t = tasks[i];
            float tsum = 0.0f;
            for(int i = 0; i < NTRIALS; i++) {
                tsum += t.metric(i, a->Number(), m);
            }
            sum += tsum / metric_tasks.size();
        }

        float mean = (sum / NTRIALS);
        float result = weight * mean;

        return result;
    }
};

#define st(x...) new ScoredTest(x)

vector<ScoredTest *> scored_tests = {
    st("test1",
       0.2f,
       {
           {Speak,  Freq,   10}
       }),

    st("test2",
       0.2f,
       {
           {Delay,  Freq,   10},
           {Speak,  Freq,   10}
       }),

    st("test3",
       0.2f,
       {
           {Delay,  Freq,   10},
           {Speak,  Freq,   10},
           {Break,  Freq,    5}
       }),

    st("test4",
       0.2f,
       {
           {Delay,  Freq,   10},
           {Speak,  Silent, 10},
           {Break,  Silent,  5}
       }),

    st("test5",
       0.2f,
       {
           {Delay,   Freq,   10},
           {Delay,   Silent,  5},
           {Speak,   Silent, 10},
           {Break,   Silent,  5}
       }),

    st("test6",
       0.2f,
       {
           {Delay,   Freq,   10},
           {Delay,   Silent, 10},
           {Speak,   Silent, 10},
           {Break,   Silent,  5}
       })
};

float compute_agent_fitness(agent *a) {
    float metric_score(agent *a, Metric m);

    float score =
        (0.6f * metric_score(a, Metric::Correspondence))
        + (0.2f * metric_score(a, Metric::Respond))
        + (0.1f * metric_score(a, Metric::Delay))
        + (0.1f * metric_score(a, Metric::Break));

    return score;
}

float metric_score(agent *a, Metric m) {
    float sum = 0.0f;

    for(ScoredTest *t: scored_tests) {
        float tmetric = t->metric(a, m);
        
        sum += tmetric;
    }

    return sum;
}

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
};

struct Test0 : public TestImpl<Test0TrialState>
{
    const long Timesteps_sound_on = 10;
    const long Timesteps_sound_off = 5;

    const long Phase0_end = Timesteps_sound_on;
    const long Phase1_end = Phase0_end + Timesteps_sound_off;

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

    virtual void end_generation(long generation_number,
                                vector<long> &ranking) {
        if(generation_number % GENERATION_LOG_FREQUENCY == 0) {
            for(long agent_number: ranking) {
                for(int i = 0; i < NTRIALS; i++) {
                    auto &trial_state = get(i, agent_number);
                    trial_state.covariance = covariance(trial_state.x, trial_state.y);
                }
            }

            // Trials
            {
                static const char *colnames[] = {
                    "Trial", "Covariance", NULL
                };
                static const datalib::Type coltypes[] = {
                    INT,     FLOAT,
                };

                char path[512];
                sprintf(path, "run/generations/%ld/test0-trials.log", generation_number);
                DataLibWriter *writer = new DataLibWriter( path, true, true );

                for(long agent_number: ranking) {
                    char tableName[32];
                    sprintf(tableName, "Agent%ld", agent_number);

                    writer->beginTable( tableName,
                                        colnames,
                                        coltypes );

                    for(int i = 0; i < NTRIALS; i++) {
                        auto &trial_state = get(i, agent_number);

                        writer->addRow( i, trial_state.covariance );
                    }

                    writer->endTable();
                }

                delete writer;
            }
        }
    }

    virtual void reset() {
        for(int i = 0; i < NTRIALS; i++) {
            fitness[i].clear();
        }
    }
};

TrialsState::TrialsState(TSimulation *sim_)
: elites(sim_->fNumberFittest, true)
{
    sim = sim_;
    generation_number = -1;
    test_number = -1;
    trial_number = -1;

    //tests.push_back(new Test0());
    for(ScoredTest *t: scored_tests) {
        tests.push_back(t);
    }
    
    sim->fMaxSteps = 0;
}

TrialsState::~TrialsState() {
}

vector<agent *> TrialsState::create_generation(size_t nagents,
                                               size_t nseeds,
                                               size_t ncrossover,
                                               FittestList &elites) {
    //db("CREATING NEW GENERATION")

    vector<agent *> agents;

    for(size_t i = 0; i < nagents; i++) {
        agent *a = agent::getfreeagent(sim, &sim->fStage);
        a->setx(0.0f);
        a->sety(0.0f);
        a->setz(0.0f);

        agents.push_back(a);
        objectxsortedlist::gXSortedObjects.add(a);
    }

    init_generation_genomes(agents,
                            nseeds,
                            ncrossover,
                            elites);

    class GrowAgents : public ITask {
    public:
        vector<agent *> &agents;

        GrowAgents(vector<agent *> &agents_) : agents(agents_) {}

        virtual void task_exec( TSimulation *sim ) {
            class GrowAgent : public ITask {
            public:
                agent *a;
                GrowAgent( agent *a ) {
                    this->a = a;
                }

                virtual void task_exec( TSimulation *sim ) {
                    a->grow( sim->fMateWait );
                }
            };

            for(agent *a: agents) {
                sim->fScheduler.postParallel(new GrowAgent(a));
            }
        }
    } growAgents(agents);

    sim->fScheduler.execMasterTask( sim,
                                    growAgents,
                                    false );

    return agents;
}

void TrialsState::init_generation_genomes(vector<agent *> &agents,
                                          size_t nseeds,
                                          size_t ncrossover,
                                          FittestList &elites) {

    size_t nagents = agents.size();
    size_t nrandom = nagents - (nseeds + ncrossover);
    size_t ninitialized = 0;

    for(size_t i = 0; i < (nseeds + nrandom); i++) {
        agent *a = agents[i];
        Genome *g = a->Genes();

        if(i < nseeds) {
            if(elites.size() == 0) {
                GenomeUtil::seed( g );
            } else {
                Genome *g_elite = elites.get(i % elites.size())->genes;
                g->copyFrom( g_elite );
            }

            g->mutate();
        } else {
            g->randomize();
        }

        ninitialized++;
    }

    if(ncrossover) {
        assert(ncrossover > 1);

        size_t iparent1 = 0;
        size_t iparent2 = 1;
        for(size_t i = (nseeds + nrandom); i < nagents; i++) {
            Genome *g = agents[i]->Genes();
            Genome *g1 = agents[iparent1]->Genes();
            Genome *g2 = agents[iparent2]->Genes();
            
            g->crossover(g1, g2, true);

            iparent2++;
            if(iparent2 == iparent1)
                iparent2++;
            if(iparent2 == (nseeds + nrandom)) {
                iparent1++;
                if(iparent1 == (nseeds + nrandom)) {
                    iparent1 = 0;
                    iparent2 = 1;
                } else {
                    iparent2 = 0;
                }
            }

            ninitialized++;
        }
    }

    assert(ninitialized == agents.size());
    for(agent *a: agents) {
        a->setGenomeReady();
    }
}

void TrialsState::timestep_begin() {
    if(generation_number == -1) {
        db("Beginning trials");
        new_generation();
    } else {
        if(sim->getStep() == trial_end_sim_step) {
            // End of trial
            if(trial_number == (NTRIALS - 1)) {
                // End of test
                if(test_number == int(tests.size() - 1)) {
                    // End of generation
                    new_generation();
                } else {
                    new_test();
                }
            } else {
                new_trial();
            }
        }
    }

    //db("Sim timestep: " << sim->getStep());

    trial_timestep++;
    
    //db("trial timestep: " << trial_timestep);

    if(trial_timestep > TEST_INTERLUDE) {
        long test_timestep = trial_timestep - TEST_INTERLUDE;

        //db("test timestep: " << test_timestep);
        int freq = freq_sequence[trial_number];
        auto test = tests[test_number];
        for(agent *a: generation_agents) {
            test->timestep_input(trial_number, test_timestep, a, freq);
        }
    }
}

void TrialsState::timestep_end() {
    if(test_number == (int)tests.size()) {
        return; // timestep_begin called end_generation()
    }

    if(trial_timestep > TEST_INTERLUDE) {
        long test_timestep = trial_timestep - TEST_INTERLUDE;
        int freq = freq_sequence[trial_number];
        auto test = tests[test_number];
        for(agent *a: generation_agents) {
            test->timestep_output(trial_number, test_timestep, a, freq);
        }
    }
}

void TrialsState::new_trial() {
    trial_number++;
    //db("*** Beginning trial " << trial_number << " of test " << test_number);

    auto test = tests[test_number];
    trial_timestep = 0;
    trial_end_sim_step = sim->getStep() + TEST_INTERLUDE + test->get_trial_timestep_count();

    for(agent *a: generation_agents) {
        a->SetEnergy(a->GetMaxEnergy());
        a->setx(0.0);
        a->setz(0.0);
        show_black(a);
    }
}

void TrialsState::new_test() {
    test_number++;
    //db("=== Beginning test " << test_number);

    trial_number = -1;
    new_trial();
}

void TrialsState::new_generation() {
    if(generation_number == -1) {
        generation_number++;
        generation_agents = create_generation(sim->fMaxNumAgents,
                                              sim->fNumberToSeed0,
                                              sim->fMaxNumAgents - sim->fInitNumAgents,
                                              elites);
    } else {
        end_generation();
        generation_number++;
        generation_agents = create_generation(sim->fMaxNumAgents,
                                              sim->fNumberToSeed,
                                              sim->fMaxNumAgents - sim->fNumberToSeed,
                                              elites);
    }

    freq_sequence.clear();
    for(int freq = 0; freq < 2; freq++) {
        for(int i = 0; i < NTRIALS/2; i++) {
            freq_sequence.push_back(freq);
        }
    }
    shuffle(freq_sequence, generation_number);

    test_number = -1;
    new_test();
}

struct Fitness {
    agent *a;
    float score;
};

void TrialsState::end_generation() {
    vector<Fitness> fitnesses;
    for(agent *a: generation_agents) {
        fitnesses.push_back({a, compute_agent_fitness(a)});
    }

    sort(fitnesses.begin(), fitnesses.end(),
         [](const Fitness &x, const Fitness &y) {
             return y.score < x.score;
         });

    db("END OF GENERATION " << generation_number << ". Top fitness = " << fitnesses.front().score);


    vector<long> ranking;
    for(auto fitness: fitnesses) {
        ranking.push_back(fitness.a->Number());
        elites.update(fitness.a, fitness.score);
    }

    if(generation_number % GENERATION_LOG_FREQUENCY == 0) {
        char cmd[512];
        sprintf(cmd, "mkdir -p run/generations/%ld", generation_number);
        int rc = system(cmd);
        if(rc != 0) {
            fprintf(stderr, "failed executing '%s'. RC=%d", cmd, rc);
            exit(rc);
        }
    }

    for(auto test: tests) {
        test->end_generation(generation_number, ranking);
    }

    if(generation_number % GENERATION_LOG_FREQUENCY == 0) {
        char path[512];
        sprintf(path, "run/generations/%ld/fitness.txt", generation_number);
        FILE *ffitness = fopen(path , "w" );

        if(generation_number == 0) {
            system("mkdir -p run/genome/Fittest");
        }

        for(int i = 0; i < sim->fNumberFittest; i++)
        {
            Fitness &fit = fitnesses[i];
            fprintf( ffitness, "%ld %f\n", fit.a->Number(), fit.score );

            {
                genome::Genome *g = fit.a->Genes();
                char path[512];
                sprintf( path, "run/genome/Fittest/genome_%ld.txt", fit.a->Number() );
                AbstractFile *out = AbstractFile::open(globals::recordFileType, path, "w");
                g->dump(out);
                delete out;
            }

        }

        fclose( ffitness );
    }

    for(agent *a: generation_agents) {
        a->Die();
        delete a;
    }

    objectxsortedlist::gXSortedObjects.clear();

    for(Test *t: tests) {
        t->reset();
    }

    if( exists("run/stop") ) {
        exit(0);
    }
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

void echo(vector<float> &x) {
    for(auto v: x)
        cout << v << " ";
    cout << endl;
}

float covariance(vector<float> &x, vector<float> &y) {
    size_t n = x.size();
    assert(y.size() == n);
    float xmean = mean(x);
    float ymean = mean(y);

    float result = 0.0f;
    for(size_t i = 0; i < n; i++) {
        result += (x[i] - xmean) * (y[i] - ymean);
    }
    result /= n;

    return result;
}

int get_max_repeat(vector<int> &x) {
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

void shuffle(vector<int> &x, int seed) {
    auto rng = std::default_random_engine(seed);

    do {
        shuffle(x.begin(), x.end(), rng);
    } while(get_max_repeat(x) > 4);
}
#endif
