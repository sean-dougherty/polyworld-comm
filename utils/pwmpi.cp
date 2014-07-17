#define INCLUDE_MPI
#include "pwmpi.h"

#include "pwassert.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <mpi.h>
#include <semaphore.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace std;

static void find_gpus();
static void acquire_gpu();

#define dbr(x...) {                                 \
        char _buf[2048];                            \
        sprintf(_buf, x);                           \
        printf("[r%2d]: %s\n", world_rank, _buf);   \
        fflush(stdout);                             \
    }

static sem_t *sem = nullptr;
static int gpu_index = -1;
static bool own_gpu = false;
static int world_rank = -1;
static int world_size = -1;

const int Tag_Worker_Send_Fittest = 0;
const int Tag_Master_Send_Fittest = 1;
const int Tag_End_Simulation = 2;

#define MIGRATION_PERIOD 1

#define MAX_NODE_RANKS 1024
#define MAX_GPUS 5

struct shared_memory_t {
    bool initialized;
    size_t nranks;
    int ranks[MAX_NODE_RANKS];

    size_t ngpus;
    struct gpu_state_t {
        int cuda_index;
        sem_t sem;
        size_t process_slots_available;
    } gpu_state[MAX_GPUS];


    sem_t bld_sem;

} *shared_memory;

namespace pwmpi {

    Worker *worker = nullptr;
    Master *master = nullptr;

    void init(int *argc, char ***argv) {
        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if(!is_master()) {
            require(-1 != nice(1));
        }

        dbr("initializing mpi...");
    
        sem_unlink("/pwmpi");
        shm_unlink("/pwmpi");

        MPI_Barrier(MPI_COMM_WORLD);

        sem = sem_open("/pwmpi",
                       O_CREAT,
                       S_IRWXU,
                       1);
        require(sem != nullptr);

        require( 0 == sem_wait(sem) );

        int shm_fd = shm_open("/pwmpi",
                              O_CREAT | O_RDWR,
                              S_IRWXU);
        require( shm_fd > -1 );
        require( ftruncate(shm_fd, sizeof(shared_memory_t)) == 0 );

        shared_memory = (shared_memory_t*)mmap(nullptr,
                                               sizeof(shared_memory_t),
                                               PROT_READ | PROT_WRITE,
                                               MAP_SHARED,
                                               shm_fd,
                                               0);
        require( shared_memory != nullptr );

        if(!shared_memory->initialized) {
            shared_memory->initialized = true;

            require( 0 == sem_init(&shared_memory->bld_sem, 1, 1) );

            find_gpus();
        }

        require( (shared_memory->nranks + 1) < MAX_NODE_RANKS );
        shared_memory->ranks[shared_memory->nranks++] = world_rank;

        acquire_gpu();

        MPI_Barrier(MPI_COMM_WORLD);

        sem_unlink("/pwmpi");
        shm_unlink("/pwmpi");

        worker = new Worker();
        if(is_master()) {
            master = new Master();
        }
    }

    void finalize() {
        if(own_gpu)
            gpu_unlock();

        MPI_Finalize();
    }

    bool is_master() {
        return world_rank == 0;
    }

    int rank() {
        return world_rank;
    }

    int size() {
        return world_size;
    }

    void bld_lock() {
        require( 0 == sem_wait(&shared_memory->bld_sem) );
    }

    void bld_unlock() {
        require( 0 == sem_post(&shared_memory->bld_sem) );
    }

    void gpu_lock() {
        assert(!own_gpu);
        dbr("Waiting on GPU lock");
        require( 0 == sem_wait(&shared_memory->gpu_state[gpu_index].sem) );
        dbr("Got GPU lock");
        own_gpu = true;
    }

    void gpu_unlock() {
        assert(own_gpu);
        require( 0 == sem_post(&shared_memory->gpu_state[gpu_index].sem) );
        own_gpu = false;
    }

    struct Message_Fittest {
        long agent_id;
        float fitness;
        int genome_len;

        static int get_message_size(int genome_len) {
            return sizeof(Message_Fittest) + genome_len;
        }

        static void alloc(unsigned char **buffer,
                          int *buffer_len,
                          int genome_len) {
            assert(*buffer == nullptr);
            int message_size = get_message_size(genome_len);
            *buffer_len = message_size;
            *buffer = (unsigned char *)malloc(message_size);
        }

        static void create(unsigned char **buffer,
                           int *buffer_len,
                           int genome_len,
                           long agent_id,
                           float fitness,
                           unsigned char *genome) {
            if(*buffer == nullptr)
                alloc(buffer, buffer_len, genome_len);

            Message_Fittest *header = (Message_Fittest *)*buffer;
            header->agent_id = agent_id;
            header->fitness = fitness;
            header->genome_len = genome_len;
        
            unsigned char *payload = (unsigned char *)(header + 1);
            memcpy(payload, genome, genome_len);
        }

        static int get_genome_len(unsigned char *buffer) {
            return ((Message_Fittest *)buffer)->genome_len;
        }

        static long get_agent_id(unsigned char *buffer) {
            return ((Message_Fittest *)buffer)->agent_id;
        }

        static float get_fitness(unsigned char *buffer) {
            return ((Message_Fittest *)buffer)->fitness;
        }

        static unsigned char *get_genome(unsigned char *buffer) {
            Message_Fittest *header = (Message_Fittest *)buffer;
            return (unsigned char *)(header + 1);
        }
    };

    Worker::Worker()
        : last_generation_sent(0)
        , send_request(MPI_REQUEST_NULL)
        , send_buffer(nullptr)
        , send_buffer_len(0)
        , last_generation_received(0)
        , recv_request(MPI_REQUEST_NULL)
        , recv_buffer(nullptr)
        , recv_buffer_len(0)
    {
    }

    void Worker::send_fittest(int generation,
                              long agent_id,
                              float fitness,
                              unsigned char *genome,
                              int genome_len) {

        // If we're still waiting for the master to receive our previous message, do nothing.
        if(send_request != MPI_REQUEST_NULL) {
            int complete;
            MPI_Test(&send_request, &complete, MPI_STATUS_IGNORE);
            if(!complete) {
                dbr("Previous send_fittest still pending. curr gen=%d, send gen=%d",
                    generation, last_generation_sent);
                return;
            }
        }

        // If we haven't received an update from the master since our last send, do nothing.
        if( last_generation_received < last_generation_sent ) {
            dbr("Haven't received fittest since last send. last_received=%d, last_sent=%d",
                last_generation_received, last_generation_sent);
            return;
        }

        // If enough generations have passed since our last update from the master,
        // proceed to send our fittest.
        if( 1 + generation - last_generation_received >= MIGRATION_PERIOD ) {
            last_generation_sent = generation;

            dbr("Sending fittest. gen=%d, agent_id=%ld, fitness=%f",
                generation, agent_id, fitness);

            Message_Fittest::create(&send_buffer,
                                    &send_buffer_len,
                                    genome_len,
                                    agent_id,
                                    fitness,
                                    genome);

            MPI_Isend(send_buffer,
                      send_buffer_len,
                      MPI_CHAR,
                      0,
                      Tag_Worker_Send_Fittest,
                      MPI_COMM_WORLD,
                      &send_request);
        } else {
            dbr("Too few generations elapsed for send_fittest. last_received=%d, curr=%d",
                last_generation_received, generation);
        }
    }

    bool Worker::recv_fittest(int generation,
                              long &agent_id,
                              float &fitness,
                              unsigned char *genome,
                              int genome_len) {
        bool received = false;

        // If we've got an outstanding receive, try to complete processing it.
        if(recv_request != MPI_REQUEST_NULL) {
            int complete;
            MPI_Test(&recv_request, &complete, MPI_STATUS_IGNORE);
            if(!complete) {
                return false;
            } else {
                received = true;
            }
            last_generation_received = generation;

            require( Message_Fittest::get_genome_len(recv_buffer) == genome_len );
            agent_id = Message_Fittest::get_agent_id(recv_buffer);
            fitness = Message_Fittest::get_fitness(recv_buffer);
            memcpy(genome, Message_Fittest::get_genome(recv_buffer), genome_len);

            dbr("RECEIVED FITTEST (gen %d): %f %ld", generation, fitness, agent_id);

        // If there's no pending receive, then this is our first time.
        } else if(recv_buffer == nullptr) {
            Message_Fittest::alloc(&recv_buffer,
                                   &recv_buffer_len,
                                   genome_len);
        }

        MPI_Irecv(recv_buffer,
                  recv_buffer_len,
                  MPI_CHAR,
                  0,
                  Tag_Master_Send_Fittest,
                  MPI_COMM_WORLD,
                  &recv_request);

        return received;
    }

    bool Worker::is_simulation_ended() {
        int test;
        MPI_Iprobe(0,
                   Tag_End_Simulation,
                   MPI_COMM_WORLD,
                   &test,
                   MPI_STATUS_IGNORE);
        return test != 0;
    }

    Master::Master()
        : send_requests(nullptr)
        , send_buffer(nullptr)
        , send_buffer_len(0)
        , recv_requests(nullptr)
        , pending_recvs_count(0)
        , recv_buffers(nullptr)
        , recv_buffer_lens(0)
    {
    }

    void Master::update_fittest(int genome_len) {
        long agent_id;
        float fitness;
        unsigned char genome[genome_len];

        if(recv_fittest(agent_id,
                        fitness,
                        genome,
                        genome_len)) {

            send_fittest(agent_id,
                         fitness,
                         genome,
                         genome_len);
        }
    }

    void Master::end_simulation() {
        char dummy;
        for(int i = 0; i < world_size; i++) {
            MPI_Send(&dummy,
                     0,
                     MPI_CHAR,
                     i,
                     Tag_End_Simulation,
                     MPI_COMM_WORLD);
        }
    }

    void Master::send_fittest(long agent_id,
                              float fitness,
                              unsigned char *genome,
                              int genome_len) {
        if(send_requests) {
            dbr("Waiting on previous sends");
            // These are all complete now. We just need to clean up the MPI resources.
            for(int i = 0; i < world_size; i++) {
                MPI_Wait(send_requests + i, MPI_STATUS_IGNORE);
            }
            dbr("Done waiting");
        } else {
            send_requests = new MPI_Request[world_size];
        }

        // We only need to create one send buffer, which we send to everyone.
        Message_Fittest::create(&send_buffer,
                                &send_buffer_len,
                                genome_len,
                                agent_id,
                                fitness,
                                genome);

        // Send to everyone via point-to-point. We can't do a broadcast
        // because optimal use of GPU requires processes executing out of
        // sync.
        for(int i = 0; i < world_size; i++) {
            MPI_Isend(send_buffer,
                      send_buffer_len,
                      MPI_CHAR,
                      i,
                      Tag_Master_Send_Fittest,
                      MPI_COMM_WORLD,
                      send_requests + i);
        }
    }

    bool Master::recv_fittest(long &agent_id,
                              float &fitness,
                              unsigned char *genome,
                              int genome_len) {
        bool received_all = false;

        if(recv_requests == nullptr) {
            recv_requests = new MPI_Request[world_size]();
            recv_buffers = new unsigned char *[world_size]();
            recv_buffer_lens = new int[world_size]();

            for(int i = 0; i < world_size; i++) {
                Message_Fittest::alloc(recv_buffers + i,
                                       recv_buffer_lens + i,
                                       genome_len);
            }
        }

        if(pending_recvs_count > 0) {
            dbr("pending_recvs_count0=%d", pending_recvs_count);

            for(int i = 0; i < pending_recvs_count; i++) {
                dbr("  %p", recv_requests[i]);
            }

            int n = pending_recvs_count;
            int i_remaining = 0;
            for(int i = 0; i < n; i++) {
                int complete;
                dbr("  testing %p", recv_requests[i]);
                MPI_Test(recv_requests + i, &complete, MPI_STATUS_IGNORE);
                if(!complete) {
                    dbr("pending");
                    recv_requests[i_remaining++] = recv_requests[i];
                } else {
                    dbr("COMPLETE");
                    pending_recvs_count--;
                }
            }
            dbr("pending_recvs_count1=%d", pending_recvs_count);
            for(int i = 0; i < pending_recvs_count; i++) {
                dbr("  %p", recv_requests[i]);
            }

            if(pending_recvs_count == 0) {
                received_all = true;
                
                float max_fitness = Message_Fittest::get_fitness(recv_buffers[0]);
                int i_max_fitness = 0;
                for(int i = 1; i < world_size; i++) {
                    float fitness = Message_Fittest::get_fitness(recv_buffers[i]);
                    if(fitness > max_fitness) {
                        max_fitness = fitness;
                        i_max_fitness = i;
                    }
                }

                agent_id = Message_Fittest::get_agent_id(recv_buffers[i_max_fitness]);
                fitness = Message_Fittest::get_fitness(recv_buffers[i_max_fitness]);
                memcpy(genome,
                       Message_Fittest::get_genome(recv_buffers[i_max_fitness]),
                       genome_len);

                dbr("RECEIVED ALL FITTEST: %f %ld", fitness, agent_id);
            }
        }

        if(pending_recvs_count == 0) {
            for(int i = 0; i < world_size; i++) {
                MPI_Irecv(recv_buffers[i],
                          recv_buffer_lens[i],
                          MPI_CHAR,
                          i,
                          Tag_Worker_Send_Fittest,
                          MPI_COMM_WORLD,
                          recv_requests + i);
            }

            pending_recvs_count = world_size;
        }

        return received_all;
    }
}

static void find_gpus() {
    int ngpus;
    require( cudaSuccess == cudaGetDeviceCount(&ngpus) );
    require( ngpus <= MAX_GPUS );

    shared_memory->ngpus = ngpus;
    for(size_t i = 0; i < shared_memory->ngpus; i++) {
        shared_memory_t::gpu_state_t &gpu = shared_memory->gpu_state[i];

        require( 0 == sem_init(&gpu.sem, 1, 1) );

        gpu.cuda_index = (int)i;
        gpu.process_slots_available = 2;
    }

    dbr("found %d gpus", ngpus);
}

static void acquire_gpu() {
    for(size_t i = 0; i < shared_memory->ngpus; i++) {
        shared_memory_t::gpu_state_t &gpu = shared_memory->gpu_state[i];
        
        if(gpu.process_slots_available > 0) {
            gpu.process_slots_available--;
            gpu_index = (int)i;
            break;
        }
    }

    dbr("acquired gpu %d", gpu_index);

    require( gpu_index > -1 );
    require( 0 == sem_post(sem) );

    require( cudaSuccess == cudaSetDevice(shared_memory->gpu_state[gpu_index].cuda_index) );
}
