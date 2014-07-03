#include "datalib.h"
#include "GenomeUtil.h"
#include "Logs.h"
#include "PathDistance.h"
#include "Retina.h"
#include "trials.h"
#include "Simulation.h"
#include "SoundPatch.h"

#include <algorithm>
#include <functional>
#include <string>

using namespace datalib;
using namespace genome;
using namespace std;

#if TRIALS

#define NDEMES 1
#define MIGRATION_PERIOD 5
#define TOURNAMENT_SIZE 5
#define ALLOW_SELF_CROSSOVER true

#define GENERATION_LOG_FREQUENCY 20

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

    virtual void log_performance(long agent_number,
                                 const char *path_dir) {
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
            sprintf(path, "%s/%s-trial-metrics.log", path_dir, name);        
            DataLibWriter *writer = new DataLibWriter( path, true, true );
            vector<Variant> colvalues;

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

            delete writer;
        }
    }

    virtual void reset() {
        for(Task &t: tasks) {
            t.reset();
        }
    }

    float metric(agent *a, Metric m) {
        vector<int> &metric_tasks = tasks_by_metric[int(m)];
        if(metric_tasks.empty())
            return weight;  // This ensures the overall metric score will sum to 1.0

        float sum = 0.0f;

        for(int i: metric_tasks) {
            Task &t = tasks[i];
            float tsum = 0.0f;
            for(int i = 0; i < NTRIALS; i++) {
                tsum += t.metric(i, a->Number(), m);
            }
            sum += tsum;
        }

        sum /= metric_tasks.size();

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

void normalize_test_weights() {
    float sum = 0.0f;
    for(auto t: scored_tests) {
        sum += t->weight;
    }

    for(auto t: scored_tests) {
        t->weight /= sum;
    }
}

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

namespace selection {
    typedef function<vector<pair<size_t, size_t>> (size_t nparents,
                                                   size_t nchildren,
                                                   function<float (size_t i)> get_fitness,
                                                   default_random_engine &rng)> method;

    vector<pair<size_t, size_t>> tournament(size_t nparents,
                                            size_t nchildren,
                                            function<float (size_t i)> get_fitness,
                                            default_random_engine &rng) {
        uniform_int_distribution<size_t> dist(0, nparents - 1);

#if ALLOW_SELF_CROSSOVER
        auto select_parent = [&dist, &rng, &get_fitness] (size_t exclude) {
            size_t winner = 0;
            float winner_fitness = 0.0f;

            for(int icandidate = 0; icandidate < TOURNAMENT_SIZE; icandidate++) {
                size_t i = dist(rng);
                float fitness = get_fitness(i);
                if(fitness > winner_fitness) {
                    winner = i;
                    winner_fitness = fitness;
                }
            }

            return winner;
        };
#else
        auto select_parent = [&dist, &rng, &get_fitness] (size_t exclude) {
            array<size_t, TOURNAMENT_SIZE + 1> excludes;
            excludes.fill(exclude);
            size_t winner = exclude;
            float winner_fitness = 0.0f;

            for(int icandidate = 0; icandidate < TOURNAMENT_SIZE; icandidate++) {
                while(true) {
                    size_t i = dist(rng);
                    if(find(excludes.begin(), excludes.end(), i) == excludes.end()) {
                        excludes[icandidate] = i;
                        float fitness = get_fitness(i);
                        if(fitness > winner_fitness) {
                            winner = i;
                            winner_fitness = fitness;
                        }
                        break;
                    }
                }
            }

            assert(winner != exclude);

            return winner;
        };
#endif

        vector<pair<size_t,size_t>> result;
        for(size_t i = 0; i < nchildren; i++) {
            size_t parent1 = select_parent(nparents);
            size_t parent2 = select_parent(parent1);

            result.emplace_back(parent1, parent2);
        }

        return result;
    }
                                                   
}

Deme::Deme(TSimulation *sim_, size_t id_, size_t nagents_, size_t nelites)
    : sim( sim_ )
    , id( id_ )
    , nagents( nagents_ )
    , elites( nelites, true)
    , prev_generation( nagents, true)
{
}

vector<agent *> Deme::create_generation(long generation_number_) {
    assert(generation_number_ == generation_number + 1);
    generation_number = generation_number_;

    rng.seed( (id + 1) * (generation_number + 1) );

    generation_agents.clear();
    for(size_t i = 0; i < nagents; i++) {
        agent *a = agent::getfreeagent(sim, &sim->fStage);
        generation_agents.push_back(a);
    }

    if(generation_number == 0) {
        init_generation0_genomes(generation_agents);
    } else {
        default_random_engine rng(generation_number);
        init_generation_genomes(generation_agents);
    }
    for(agent *a: generation_agents) {
        a->setGenomeReady();
    }
    
    return generation_agents;
}

void Deme::init_generation0_genomes(vector<agent *> &agents) {
    for(agent *a: agents) {
        a->Genes()->randomize();
    }
}

void Deme::init_generation_genomes(vector<agent *> &next_generation) {
    size_t nparents = size_t(prev_generation.size()) + size_t(elites.size());
    size_t nchildren = next_generation.size();

    auto get_fitness = [=] (size_t i) {
        if(i < size_t(elites.size()))
            return elites.get(i)->fitness;
        else
            return prev_generation.get(i - elites.size())->fitness;
    };
    
    auto get_parent = [=] (size_t i) {
        if(i < size_t(elites.size()))
            return elites.get(i)->genes.get();
        else
            return prev_generation.get(i - elites.size())->genes.get();
    };
    
    vector<pair<size_t,size_t>> parents = selection::tournament(nparents,
                                                                nchildren,
                                                                get_fitness,
                                                                rng);
    assert(parents.size() == nchildren);

    for(size_t i = 0; i < nchildren; i++) {
        pair<size_t, size_t> p = parents[i];
        
        next_generation[i]->Genes()->crossover(get_parent(p.first),
                                               get_parent(p.second),
                                               true);
    }
}

void Deme::end_generation() {
    prev_generation.clear();
    for(agent *a: generation_agents) {
        prev_generation.update(a, compute_agent_fitness(a));
    }

    for(int i = 0; i < prev_generation.size(); i++) {
        FitStruct *fs = prev_generation.get(i);
        if( elites.update(fs) < 0 ) {
            break;
        }
    }

    generation_agents.clear();
}

FitStruct *Deme::get_fittest() {
    return prev_generation.get(0);
}

void Deme::accept_immigrant(FitStruct *fs) {
    prev_generation.dropLast();
    prev_generation.update(fs);
}

TrialsState::TrialsState(TSimulation *sim_)
    : sim( sim_ )
    , elites( 1, true )
    , prev_generation( NDEMES, true )
{
    sim = sim_;

    normalize_test_weights();

    for(ScoredTest *t: scored_tests) {
        tests.push_back(t);
    }
    
    sim->fMaxSteps = 0;

    for(size_t i = 0; i < NDEMES; i++) {
        demes.push_back( new Deme(sim,
                                  i,
                                  sim->fMaxNumAgents / NDEMES,
                                  sim->fNumberFittest / NDEMES) );
    }
}

TrialsState::~TrialsState() {
}

vector<agent *> TrialsState::create_generation() {
    db("CREATING NEW GENERATION");

    vector<agent *> agents;
    for(Deme *deme: demes) {
        for(agent *a: deme->create_generation(generation_number)) {
            a->setx(0.0f);
            a->sety(0.0f);
            a->setz(0.0f);

            agents.push_back(a);
            objectxsortedlist::gXSortedObjects.add(a);
        }
    }

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
    if(generation_number != -1) {
        end_generation();
    }
    generation_number++;
    generation_agents = create_generation();

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

void log_fitness(const string &path,
                 int nagents,
                 function<FitStruct *(int i)> get_fitness) {
    FILE *ffitness = fopen(path.c_str() , "w" );

    for(int i = 0; i < nagents; i++) {
        FitStruct *fs = get_fitness(i);
        fprintf( ffitness, "%ld %f\n", fs->agentID, fs->fitness );
    }

    fclose(ffitness);
}

void log_fitness(const string &path,
                 FittestList &fittest) {
    log_fitness(path,
                fittest.size(),
                [&fittest] (int i) { return fittest.get(i); });
}

void log_genome(FitStruct *fs) {
    char path[512];

    sprintf( path, "run/genome/Fittest/genome_%ld.txt", fs->agentID );
    if( !AbstractFile::exists(path) ) {
        makeParentDir(path);

        AbstractFile *out = AbstractFile::open(globals::recordFileType, path, "w");
        fs->genes->dump(out);
        delete out;
    }
}

void TrialsState::log_elite(FitStruct *fs) {
    char path_dir[512];
    sprintf(path_dir, "run/elites/%ld", fs->agentID);

    makeDirs( path_dir );

    log_fitness( string(path_dir) + "/fitness.txt",
                 1,
                 [fs] (int i) { return fs; });
    log_genome( fs );

    for(Test *t: tests) {
        t->log_performance(fs->agentID, path_dir);
    }
}

void TrialsState::end_generation() {
    db("END OF GENERATION " << generation_number);

    prev_generation.clear();
    for(Deme *deme: demes) {
        deme->end_generation();
        prev_generation.update( deme->get_fittest() );
    }

    for(int i = 0; i < prev_generation.size(); i++) {
        FitStruct *fs = prev_generation.get(i);
        if( elites.update(fs) >= 0 ) {
            log_elite(fs);
        } else {
            break;
        }
    }

    db("  Generation fitness = " << prev_generation.get(0)->fitness);
    if(elites.size() > 0) {
        db("      Global fitness = " << elites.get(0)->fitness);
    }

    if(generation_number % GENERATION_LOG_FREQUENCY == 0) {
        char path_dir[512];
        sprintf(path_dir, "run/generations/%ld", generation_number);
        makeDirs(path_dir);

        if(elites.size() > 0) {
            char path[512];
            sprintf(path, "%s/global-fitness.txt", path_dir);
            log_fitness(path, elites);
        }
        {
            char path[512];
            sprintf(path, "%s/generation-fitness.txt", path_dir);
            log_fitness(path, prev_generation);
        }
    }

    for(agent *a: generation_agents) {
        a->Die();
        logs->postEvent( AgentDeathEvent(a, LifeSpan::DR_NATURAL) );
        delete a;
    }

    objectxsortedlist::gXSortedObjects.clear();

    for(Test *t: tests) {
        t->reset();
    }

    if( generation_number == 5 ) {
        sim->End("DEBUG BRAINS");
    }

    if( ((generation_number + 1) % MIGRATION_PERIOD) == 0) {
        FitStruct *immigrant = prev_generation.get(0);
        db("PERFORMING MIGRATION (" << immigrant->agentID << " " << immigrant->fitness << ")");

        for(Deme *deme: demes) {
            deme->accept_immigrant( immigrant );
        }
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
