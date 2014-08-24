#include "datalib.h"
#include "GenomeUtil.h"
#include "Logs.h"
#include "PathDistance.h"
#include "pwmpi.h"
#include "Retina.h"
#include "trials.h"
#include "RandomNumberGenerator.h"
#include "Simulation.h"
#include "SoundPatch.h"
#include "timer.h"

#include <algorithm>
#include <functional>
#include <string>

using namespace datalib;
using namespace genome;
using namespace std;

#if TRIALS

#define NTRIALS 24
//#define MAX_GENERATIONS 5
#define EPSILON 0.00001f
#define MAX_FITNESS 1.0f
#define MAX_NDEMES 1
#define ELITES_PER_DEME 5
#define MIGRATE_ELITES false
#define TOURNAMENT_SIZE 5
#define ALLOW_SELF_CROSSOVER true
#define SERIAL_GENOME true
#define SEQUENCE_LENGTH 2
#define SEQ_TEST_ALL 1
#define SEQ_TEST_FIRST 2
#define SEQ_TEST_LAST 3
#define SEQ_TEST_MODE SEQ_TEST_ALL
#define INTERLUDE_SHOW_BLUE true

#define HIGH_MUTATE_NO_PROGRESS 0
#define HIGH_MUTATE_RATE_MULTIPLIER 10.0f;


#define GENERATION_LOG_FREQUENCY 20

#define DEBUG true
#define VERBOSE false
#if DEBUG
#define db(x...) cout << x << endl;

template<typename T>
void dbvec(string msg, vector<T> &x) {
    cout << msg << ": ";
    for(auto v: x)
        cout << v << " ";
    cout << endl;
}
#else
#define db(x...)
#define dbvec(x...)
#endif

#if VERBOSE
#define vdb(x...) db(x)
#define vdbvec(x...) dbvec(x)
#else
#define vdb(x...)
#define vdbvec(x...)
#endif

TrialsState *trials = nullptr;

template<typename T>
T mean(vector<T> scores);
template<typename T>
T stddev(vector<T> scores);
void shuffle(vector<int> &x, int seed);

void show_color(agent *a, float r, float g, float b) {
    a->GetRetina()->force_color(r, g, b);
}

void show_red(agent *a) {
    vdb("showing red");
    show_color(a, 1.0, 0.0, 0.0);
}

void show_green(agent *a) {
    vdb("showing green");
    show_color(a, 0.0, 1.0, 0.0);
}

void show_blue(agent *a) {
    vdb("showing blue");
    show_color(a, 0.0, 0.0, 1.0);
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

    size_t ntrials;
    size_t nagents;

    struct AgentTrialState {
        long silence_count;
        long correspondence_count;
        long respond_count;
    };
    AgentTrialState *agent_trial_states;

    inline AgentTrialState &get_state(int trial_number, agent *a) {
        return agent_trial_states[trial_number*nagents + a->Index()];
    }

    inline int seq_index(long t_task) {
#if SEQ_TEST_MODE == SEQ_TEST_FIRST
        if(category == Speak) {
            return 0;
        }
#elif SEQ_TEST_MODE == SEQ_TEST_LAST
        if(category == Speak) {
            return SEQUENCE_LENGTH - 1;
        }
#elif SEQ_TEST_MODE != SEQ_TEST_ALL
        #error invalid mode
#endif

        return ((t_task-1) * SEQUENCE_LENGTH) / timesteps;
    }

    map<long, AgentTrialState> agent_trial_state[NTRIALS];

    void init() {
        ntrials = 0;
        nagents = 0;
        agent_trial_states = nullptr;
    }

    void init(size_t ntrials_, size_t nagents_) {
        if( !agent_trial_states
            || (nagents_ > nagents)
            || (ntrials_ > ntrials) ) {

            delete [] agent_trial_states;
            agent_trial_states = new AgentTrialState[nagents_ * ntrials_];
        }
        ntrials = ntrials_;
        nagents = nagents_;

        memset(agent_trial_states, 0, nagents * ntrials * sizeof(AgentTrialState));
    }

    void timestep_input(long t_task, agent *a, const vector<int> &seq) {
        int i = seq_index(t_task);
        int freq = seq[i];
        vdb("task timestep: " << t_task);

        //
        // Sound
        //
        if(sound == Freq) {
            vdb("playing freq " << freq);
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
            switch(i) {
            case 0:
                show_red(a);
                break;
            case 1:
                show_green(a);
                break;
            default:
                panic();
            }
            break;
        default:
            panic();
        }
    }

    void timestep_output(int trial_number,
                         long t_trial,
                         agent *a,
                         const vector<int> &seq) {
        AgentTrialState &state = get_state(trial_number, a);
        int i = seq_index(t_trial);
        int freq = seq[i];
        vdb("expecting freq " << freq);

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
            panic();
            break;
        }

        t_trial++;
    }

    float metric(int trial_number, agent *a, Metric m) {
        AgentTrialState &state = get_state(trial_number, a);

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
            panic();
        }
    }
};

struct ScoredTest : public Test {
    const char *name;
    float weight;
    vector<Task> tasks;
    Task *task = nullptr;
    vector<int> tasks_by_metric[int(Metric::__N)];
    long task_timestep;
    long timesteps;

ScoredTest(const char *name_,
           float weight_,
           vector<Task> tasks)
        : name(name_)
        , weight(weight_)
        , tasks(tasks)
    {
        timesteps = 0;
        for(auto &t: tasks) {
            t.init();
            timesteps += t.timesteps;
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

    virtual void init(size_t ntrials, size_t nagents) {
        for(Task &t: tasks) {
            t.init(ntrials, nagents);
        }

        task = tasks.data();
        task_timestep = 1;
    }

    virtual long get_trial_timestep_count() {
        return timesteps;
    }

    virtual void timestep_begin(long test_timestep) {
        if(test_timestep == 1) {
            task = tasks.data();            
            task_timestep = 1;
        } else {
            task_timestep++;
            if(task_timestep > task->timesteps) {
                task++;
                task_timestep = 1;
            }
        }

        assert( size_t(task - tasks.data()) < tasks.size() );
    }

    virtual void timestep_input(int trial_number,
                                long test_timestep,
                                agent *a,
                                const vector<int> &freq) {
        task->timestep_input(task_timestep, a, freq);
    }

    virtual void timestep_output(int trial_number,
                                 long test_timestep,
                                 agent *a,
                                 const vector<int> &freq) {
        
        task->timestep_output(trial_number, task_timestep, a, freq);
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
                tsum += t.metric(i, a, m);
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
    st("test6",
       0.2f,
       {
           {Delay,   Freq,   10 * SEQUENCE_LENGTH},
           {Delay,   Silent, 10},
           {Speak,   Silent, 10 * SEQUENCE_LENGTH},
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
    , elites( nelites, true )
    , prev_generation( nagents, true)
{
    require(nagents > 0);
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
        init_generation_genomes(generation_agents);
    }
    for(agent *a: generation_agents) {
        a->setGenomeReady();
    }
    
    return generation_agents;
}

void Deme::init_generation0_genomes(vector<agent *> &agents) {
#if SERIAL_GENOME
    for(agent *a: agents) {
        a->Genes()->randomize();
    }
#else
#pragma omp parallel for
    for(size_t i = 0; i < agents.size(); i++) {
        agents[i]->Genes()->randomize();
    }
#endif
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

    float mutation_rate_multiplier = 1.0f;
#if HIGH_MUTATE_NO_PROGRESS > 0
    if( (no_progress >= HIGH_MUTATE_NO_PROGRESS) && ((generation_number - high_mutate_generation) >= HIGH_MUTATE_NO_PROGRESS) ) {
        db("Using high mutation rate, generation=" << generation_number << ", no_progress=" << no_progress << ", high_mutate_generation=" << high_mutate_generation);

        high_mutate_generation = generation_number;
        mutation_rate_multiplier = HIGH_MUTATE_RATE_MULTIPLIER;
    }
#endif

#if SERIAL_GENOME
    for(size_t i = 0; i < nchildren; i++) {
        pair<size_t, size_t> p = parents[i];
        
        next_generation[i]->Genes()->crossover(get_parent(p.first),
                                               get_parent(p.second),
                                               true,
                                               mutation_rate_multiplier);
    }
#else
#pragma omp parallel for
    for(size_t i = 0; i < nchildren; i++) {
        pair<size_t, size_t> p = parents[i];
        
        next_generation[i]->Genes()->crossover(get_parent(p.first),
                                               get_parent(p.second),
                                               true,
                                               mutation_rate_multiplier);
    }
#endif
}

void Deme::end_generation() {
    prev_generation.clear();
    for(agent *a: generation_agents) {
        prev_generation.update(a, compute_agent_fitness(a));
    }

    for(int i = 0; i < prev_generation.size(); i++) {
        FitStruct *fs = prev_generation.get(i);
        if(elites.update(fs) < 0) {
            break;
        }
    }

    generation_agents.clear();

    float gen_best_score = prev_generation.get(0)->fitness;
    if(gen_best_score <= best_score) {
        no_progress++;
    } else {
        best_score = gen_best_score;
        no_progress = 0;
    }

    db("no_progress = " << no_progress << ", gen_best_score = " << gen_best_score << ", best_score = " << best_score);
}

FitStruct *Deme::get_fittest() {
#if MIGRATE_ELITES && (ELITES_PER_DEME > 0)
    return elites.get(0);
#else
    return prev_generation.get(0);
#endif
}

void Deme::accept_immigrant(FitStruct *fs) {
    prev_generation.dropLast();
    prev_generation.update(fs);
}

TrialsState::TrialsState(TSimulation *sim_)
    : sim( sim_ )
    , ndemes( pwmpi::get_demes_count(MAX_NDEMES) )
    , agents_per_deme( sim->fMaxNumAgents / ndemes )
    , elites( 1, true )
    , prev_generation( ndemes + 1, true )
    , genome_len( GenomeUtil::schema->getMutableSize() )
{
    sim = sim_;

#if !SERIAL_GENOME
    RandomNumberGenerator::set( RandomNumberGenerator::GENOME,
                                RandomNumberGenerator::LOCAL );
#endif

    normalize_test_weights();

    for(ScoredTest *t: scored_tests) {
        tests.push_back(t);
    }
    
    sim->fMaxSteps = 0;

    for(int i = 0; i < ndemes; i++) {
        demes.push_back( new Deme(sim,
                                  i,
                                  agents_per_deme,
                                  ELITES_PER_DEME) );
    }
}

TrialsState::~TrialsState() {
}

vector<agent *> TrialsState::create_generation() {
    db("CREATING NEW GENERATION");

    vector<agent *> agents;
    agents.resize(ndemes * agents_per_deme);

    for(int i = 0; i < ndemes; i++) {
        Deme *deme = demes[i];
        vector<agent *> deme_agents = deme->create_generation(generation_number);
        assert(deme_agents.size() == size_t(agents_per_deme));

        for(long j = 0; j < agents_per_deme; j++) {
            agent *a = deme_agents[j];

            a->setx(0.0f);
            a->sety(0.0f);
            a->setz(0.0f);

            agents[i*agents_per_deme + j] = a;
        }
    }

    for(agent *a: agents) {
        objectxsortedlist::gXSortedObjects.add(a);
    }

#pragma omp parallel for
    for(size_t i = 0; i < agents.size(); i++) {
        agents[i]->grow( sim->fMateWait );
    }

    return agents;
}

bool TrialsState::timestep_begin() {
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
                    if(!new_generation()) {
                        return false;
                    }
                } else {
                    new_test();
                }
            } else {
                new_trial();
            }
        }
    }

    vdb("Sim timestep: " << sim->getStep());

    trial_timestep++;
    
    vdb("trial timestep: " << trial_timestep);

    if(trial_timestep > TEST_INTERLUDE) {
        long test_timestep = trial_timestep - TEST_INTERLUDE;

        vdb("test timestep: " << test_timestep);
        vector<int> &seq = sequences[trial_number];
        vdbvec("seq", seq);
        auto test = tests[test_number];

        test->timestep_begin(test_timestep);

#pragma omp parallel for
        for(size_t i = 0; i < generation_agents.size(); i++) {
            test->timestep_input(trial_number, test_timestep, generation_agents[i], seq);
        }
    } else {
#if INTERLUDE_SHOW_BLUE
        for(size_t i = 0; i < generation_agents.size(); i++) {
            show_blue(generation_agents[i]);
        }
#endif
    }

    return true;
}

void TrialsState::timestep_end() {
    if(test_number == (int)tests.size()) {
        return; // timestep_begin called end_generation()
    }

    if(trial_timestep > TEST_INTERLUDE) {
        long test_timestep = trial_timestep - TEST_INTERLUDE;
        vector<int> seq = sequences[trial_number];
        auto test = tests[test_number];

#pragma omp parallel for
        for(size_t i = 0; i < generation_agents.size(); i++) {
            test->timestep_output(trial_number, test_timestep, generation_agents[i], seq);
        }
    }
}

void TrialsState::new_trial() {
    trial_number++;
    vdb("*** Beginning trial " << trial_number << " of test " << test_number);

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
    vdb("=== Beginning test " << test_number);

    trial_number = -1;
    new_trial();
}

static FitStruct create_fit(long agent_id,
                            float fitness,
                            unsigned char *genome_data,
                            int genome_len) {
    FitStruct fs;
    fs.genes = GenomeUtil::createGenome();
    fs.agentID = agent_id;
    fs.fitness = fitness;
    memcpy(fs.genes->mutable_data, genome_data, genome_len);
    return fs;
}

bool TrialsState::new_generation() {
    static double prev_start = 0.0;
    double start = seconds();

    if(generation_number != -1) {
        pwmpi::gpu_unlock();

        end_generation();

        if(pwmpi::is_master()) {
#ifdef MAX_GENERATIONS
            if( generation_number == MAX_GENERATIONS ) {
                db("REACHED MAX GENERATION");
                return false;
            }
#endif

#ifdef MAX_FITNESS
            if( elites.get(0)->fitness >= (MAX_FITNESS - EPSILON) ) {
                db("ACHIEVED MAX FITNESS: " << elites.get(0)->fitness << " " << elites.get(0)->agentID);
                return false;
            }
#endif
        }

        cout << "time to execute previous generation = " << start - prev_start << endl;
        extern double brain_time;
        static vector<double> brain_times;
        brain_times.push_back(brain_time);
        cout << "time to execute brains = " << brain_time << ", mean = " << mean(brain_times) << ", stddev = " << stddev(brain_times) << endl;
        brain_time = 0.0;
    }
    prev_start = start;

    for(Test *t: tests) {
        t->init(NTRIALS, ndemes * agents_per_deme);
    }

    generation_number++;
    
    generation_agents = create_generation();

    cout << "time to make new generation = " << seconds() - start << endl;

    const vector<vector<int>>seq_library = {
#if SEQUENCE_LENGTH == 1
        {0}, {1}, {2}
#elif SEQUENCE_LENGTH == 2
        {0,1}, {0,2}, {1,0}, {1,2}, {2,0}, {2,1}
#else
        #error Only seq len of 1 or 2 supported
#endif
    };

    sequences.clear();
    for(size_t seq_index = 0; seq_index < seq_library.size(); seq_index++) {
        for(size_t i = 0; i < (NTRIALS/seq_library.size()); i++) {
            sequences.push_back(seq_library[seq_index]);
        }
    }
    require(sequences.size() == NTRIALS);
    //shuffle(sequences, generation_number);
    {
        auto rng = std::default_random_engine(generation_number);
        shuffle(sequences.begin(), sequences.end(), rng);
    }

    test_number = -1;
    new_test();

    pwmpi::gpu_lock();

    return true;
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

    sprintf( path, "genome/Fittest/genome_%ld.txt", fs->agentID );
    if( !AbstractFile::exists(path) ) {
        makeParentDir(path);

        AbstractFile *out = AbstractFile::open(globals::recordFileType, path, "w");
        fs->genes->dump(out);
        delete out;
    }
}

void TrialsState::log_elite(FitStruct *fs) {
    char path_dir[512];
    sprintf(path_dir, "elites/%ld", fs->agentID);

    makeDirs( path_dir );

    log_fitness( string(path_dir) + "/fitness.txt",
                 1,
                 [fs] (int i) { return fs; });
    log_genome( fs );
}

void TrialsState::end_generation() {
    db("END OF GENERATION " << generation_number);

    prev_generation.clear();
    for(Deme *deme: demes) {
        deme->end_generation();
        prev_generation.update( deme->get_fittest() );
    }

    {
        FitStruct *fs = prev_generation.get(0);
        pwmpi::worker->send_fittest(generation_number,
                                    fs->agentID,
                                    fs->fitness,
                                    fs->genes->mutable_data,
                                    fs->genes->nbytes);

        
    }

    if( pwmpi::is_master() ) {
        pwmpi::master->update_fittest(genome_len);
    }

    {
        long agent_id;
        float fitness;
        unsigned char genome[genome_len];

        if( pwmpi::worker->recv_fittest(generation_number,
                                        agent_id,
                                        fitness,
                                        genome,
                                        genome_len) ) {
            FitStruct fs = create_fit(agent_id, fitness, genome, genome_len);

            prev_generation.update( &fs );
            for(Deme *deme: demes) {
                deme->accept_immigrant( &fs );
            }
        } else if(ndemes > 1) {
            FitStruct *fs = prev_generation.get(0);
            db("performing local migration: " << fs->agentID << " " << fs->fitness);
            for(Deme *deme: demes) {
                deme->accept_immigrant( fs );
            }
        }
    }

    if( pwmpi::is_master() ) {
        for(int i = 0; i < prev_generation.size(); i++) {
            FitStruct *fs = prev_generation.get(i);
            if( elites.update(fs) >= 0 ) {
                log_elite(fs);
            } else {
                break;
            }
        }

        db("  Generation fitness = " << prev_generation.get(0)->fitness);
        db("      Global fitness = " << elites.get(0)->fitness);

        if(generation_number % GENERATION_LOG_FREQUENCY == 0) {
            char path_dir[512];
            sprintf(path_dir, "generations/%ld", generation_number);
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
    }

    for(agent *a: generation_agents) {
        a->Die();
        logs->postEvent( AgentDeathEvent(a, LifeSpan::DR_NATURAL) );
        delete a;
    }

    objectxsortedlist::gXSortedObjects.clear();
}

template<typename T>
T mean(vector<T> scores) {
    T sum = 0.0;
    for(auto x: scores)
        sum += x;
    return sum / scores.size();
}

template<typename T>
T stddev(vector<T> scores) {
    const T N = scores.size();
    T sum = 0.0;
    T sum2 = 0.0;
    for(auto x: scores) {
        sum += x;
        sum2 += x*x;
    }
    T result = sqrt( (sum2 - (sum*sum) / N) / N );
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
