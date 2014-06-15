#include "datalib.h"
#include "PathDistance.h"
#include "Retina.h"
#include "trials.h"
#include "Simulation.h"
#include "SoundPatch.h"

#include <algorithm>
#include <string>

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
float covariance(vector<float> &x, vector<float> &y);
void shuffle(vector<int> &x);

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
    Wait,
    Speak
};

enum Sound {
    Freq,
    Silent
};

enum Metric {
    Delay,
    Corr,
    Resp,
    Break
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
        case Wait:
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
        case Wait:
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

    float metric(int trial_number, agent *a, Metric m) {
        AgentTrialState &state = agent_trial_state[trial_number][a->Number()];

        switch(m) {
        case Resp:
            assert(category == Speak);
            return float(state.respond_count) / timesteps;
        case Corr:
            assert(category == Speak);
            return float(state.correspondence_count) / timesteps;
        default:
            assert(false);
        }
    }

    float metric(int trial_number, agent *a) {
        AgentTrialState &state = agent_trial_state[trial_number][a->Number()];

        assert(category == Wait);

        return float(state.silence_count) / timesteps;
    }
};

struct ScoredTest : public Test {
    float weight;
    vector<Task> tasks;
    vector<long> task_ends;

    ScoredTest(float weight_,
               const vector<Task> &tasks_)
        : weight(weight_)
        , tasks(tasks)
    {
        long end = 0;

        for(auto t: tasks) {
            end += t.timesteps;
            task_ends.push_back(end);
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
            }
        }
    }

    virtual void end_generation(std::vector<long> &ranking) {
        assert(false);
    }

    float metric(agent *a, int task_index, Metric m) {
        float sum = 0.0f;
        for(int i = 0; i < NTRIALS; i++) {
            sum += tasks[task_index].metric(i, a, m);
        }
        return weight * (sum / NTRIALS);
    }

    float metric(agent *a, int task_index) {
        float sum = 0.0f;
        for(int i = 0; i < NTRIALS; i++) {
            sum += tasks[task_index].metric(i, a);
        }
        return weight * (sum / NTRIALS);
    }
};

const int TrialsState::ElitesCount = 10;

ScoredTest test1(
    0.1f,
    {
        {Speak,  Freq,   10}
    }
);

ScoredTest test2(
    0.1f,
    {
        {Wait,   Freq,   10},
        {Speak,  Freq,   10}
    }
);

ScoredTest test3(
    0.1f,
    {
        {Wait,   Freq,   10},
        {Speak,  Freq,   10},
        {Wait,   Freq,    5}
    }
);

ScoredTest test4(
    0.3f,
    {
        {Wait,   Freq,   10},
        {Speak,  Silent, 10},
        {Wait,   Silent,  5}
    }
);

ScoredTest test5(
    0.4f,
    {
        {Wait,   Freq,   10},
        {Wait,   Silent,  5},
        {Speak,  Silent, 10},
        {Wait,   Silent,  5}
    }
);

float compute_agent_fitness(agent *a) {
    float correspondence_score =
        0.6f * (
            test1.metric(a, 0, Corr)
            + test2.metric(a, 1, Corr)
            + test3.metric(a, 1, Corr)
            + test4.metric(a, 1, Corr)
            + test5.metric(a, 2, Corr)
            );

    float respond_score =
        0.2f * (
            test1.metric(a, 0, Resp)
            + test2.metric(a, 1, Resp)
            + test3.metric(a, 1, Resp)
            + test4.metric(a, 1, Resp)
            + test5.metric(a, 2, Resp)
            );

    float delay_score = 
        0.1f * (
            test2.metric(a, 0)
            + test3.metric(a, 0)
            + test4.metric(a, 0)
            + 0.5f*test5.metric(a, 0) + 0.5f*test5.metric(a, 1)
            );

    float break_score =
        0.1f * (
            test3.metric(a, 2)
            + test4.metric(a, 2)
            + test5.metric(a, 3)
            );

    return correspondence_score
        + respond_score
        + delay_score
        + break_score;
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

    virtual void end_generation(vector<long> &ranking) {
        // Trials
        {
            static const char *colnames[] = {
                "Trial", "Covariance", NULL
            };
            static const datalib::Type coltypes[] = {
                INT,     FLOAT,
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

                    writer->addRow( i, trial_state.covariance );
                }

                writer->endTable();
            }

            delete writer;
        }
    }
};

TrialsState::TrialsState(TSimulation *sim_) {
    sim = sim_;
    test_number = -1;
    trial_number = -1;

    tests.push_back(new Test0());
    tests.push_back(&test1);
    tests.push_back(&test2);
    tests.push_back(&test3);
    tests.push_back(&test4);
    tests.push_back(&test5);
    
    long nsteps = 0;
    for(auto test: tests) {
        nsteps += NTRIALS * (test->get_trial_timestep_count() + TEST_INTERLUDE);
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

void TrialsState::timestep_begin() {
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

    trial_timestep++;

    if(trial_timestep > TEST_INTERLUDE) {
        long test_timestep = trial_timestep - TEST_INTERLUDE;
        int freq = freq_sequence[trial_number];
        auto test = tests[test_number];
        for(agent *a: get_agents()) {
            test->timestep_input(trial_number, test_timestep, a, freq);
        }
    }
}

void TrialsState::timestep_end() {
    if(trial_timestep > TEST_INTERLUDE) {
        long test_timestep = trial_timestep - TEST_INTERLUDE;
        int freq = freq_sequence[trial_number];
        auto test = tests[test_number];
        for(agent *a: get_agents()) {
            test->timestep_output(trial_number, test_timestep, a, freq);
        }
    }
}

void TrialsState::init_test() {
    db("=== Beginning test " << test_number);
}

void TrialsState::end_test() {
    db("=== Ending test " << test_number);
}

void TrialsState::init_trial() {
    db("*** Beginning trial " << trial_number << " of test " << test_number);

    auto test = tests[test_number];
    trial_timestep = 0;
    trial_end_sim_step = sim->getStep() + TEST_INTERLUDE + test->get_trial_timestep_count();

    for(agent *a: get_agents()) {
        a->SetEnergy(a->GetMaxEnergy());
        a->setx(0.0);
        a->setz(0.0);
        show_black(a);
    }
}

void TrialsState::end_trial() {
    db("*** Ending trial " << trial_number << " of test " << test_number);
}

struct Fitness {
    agent *a;
    float score;
};

void TrialsState::end_generation() {
    db("END OF GENERATION");
    sim->End("endTests");

    vector<Fitness> fitnesses;
    for(agent *a: get_agents()) {
        fitnesses.push_back({a, compute_agent_fitness(a)});
    }

    sort(fitnesses.begin(), fitnesses.end(),
         [](const Fitness &x, const Fitness &y) {
             return y.score < x.score;
         });

    vector<long> ranking;
    for(auto fitness: fitnesses) {
        ranking.push_back(fitness.a->Number());
    }
    for(auto test: tests) {
        test->end_generation(ranking);
    }

    system("mkdir -p run/genome/Fittest");
    {
        FILE *ffitness = fopen( "run/genome/Fittest/fitness.txt", "w" );

        for(int i = 0; i < ElitesCount; i++)
        {
            Fitness &fit = fitnesses[i];
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

}

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

void shuffle(vector<int> &x) {
    ifstream in("trial_seed.txt");
    int seed;
    in >> seed;
    db("Food sequence RNG seed: " << seed);

    auto rng = std::default_random_engine(seed);

    do {
        shuffle(x.begin(), x.end(), rng);
    } while(get_max_repeat(x) > 4);
}
#endif
