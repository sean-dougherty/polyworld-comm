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

static bool mpi_mode = false;

static sem_t *sem = nullptr;
static int gpu_index = -1;
static int world_rank = -1;
static int world_size = -1;

const int Tag_Worker_Send_Fittest = 0;
const int Tag_Master_Send_Fittest = 1;

#define MIGRATION_PERIOD 5

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

    void init(int *argc, char ***argv) {
        mpi_mode = true;

        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        printf("rank=%d, size=%d\n", world_rank, world_size);
    
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
    }

    void finalize() {
        MPI_Finalize();
    }

    bool is_mpi_mode() {
        return mpi_mode;
    }

    bool is_master() {
        return world_rank == 0;
    }

    int rank() {
        return world_rank;
    }

    void bld_lock() {
        if(mpi_mode) {
            require( 0 == sem_wait(&shared_memory->bld_sem) );
        }
    }

    void bld_unlock() {
        if(mpi_mode) {
            require( 0 == sem_post(&shared_memory->bld_sem) );
        }
    }

    void gpu_lock() {
        if(mpi_mode) {
            printf("waiting on gpu sempahore..."); fflush(stdout);
            require( 0 == sem_wait(&shared_memory->gpu_state[gpu_index].sem) );
            printf(" OK!\n"); fflush(stdout);
        }
    }

    void gpu_unlock() {
        if(mpi_mode) {
            printf("posting gpu sempahore..."); fflush(stdout);
            require( 0 == sem_post(&shared_memory->gpu_state[gpu_index].sem) );
            printf(" OK!\n"); fflush(stdout);
        }
    }

    struct Message_Fittest {
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
                           float fitness,
                           unsigned char *genome) {
            if(*buffer == nullptr)
                alloc(buffer, buffer_len, genome_len);

            Message_Fittest *header = (Message_Fittest *)*buffer;
            header->fitness = fitness;
            header->genome_len = genome_len;
        
            unsigned char *payload = (unsigned char *)(header + 1);
            memcpy(payload, genome, genome_len);
        }

        static int get_genome_len(unsigned char *buffer) {
            return ((Message_Fittest *)buffer)->genome_len;
        }

        static float get_fitness(unsigned char *buffer) {
            return ((Message_Fittest *)buffer)->fitness;
        }

        static unsigned char *get_genome(unsigned char *buffer) {
            Message_Fittest *header = (Message_Fittest *)buffer;
            return (unsigned char *)(header + 1);
        }
    };


    void Worker::send_fittest(int generation,
                              float fitness,
                              unsigned char *genome,
                              int genome_len) {

        // If we're still waiting for the master to receive our previous message, do nothing.
        if(send_request != MPI_REQUEST_NULL) {
            int complete;
            MPI_Test(&send_request, &complete, MPI_STATUS_IGNORE);
            if(!complete)
                return;
        }

        // If we haven't received an update from the master since our last send, do nothing.
        if( last_generation_received < last_generation_sent )
            return;

        // If enough generations have passed since our last update from the master,
        // proceed to send our fittest.
        if( 1 + generation - last_generation_received >= MIGRATION_PERIOD ) {
            last_generation_sent = generation;

            Message_Fittest::create(&send_buffer,
                                    &send_buffer_len,
                                    genome_len,
                                    fitness,
                                    genome);

            MPI_Isend(send_buffer,
                      send_buffer_len,
                      MPI_CHAR,
                      0,
                      Tag_Worker_Send_Fittest,
                      MPI_COMM_WORLD,
                      &send_request);
        }
    }

    bool Worker::recv_fittest(int generation,
                              float *fitness,
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
            *fitness = Message_Fittest::get_fitness(recv_buffer);
            memcpy(genome, Message_Fittest::get_genome(recv_buffer), genome_len);

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

    bool Master::update_fittest(float *fitness,
                                unsigned char *genome,
                                int genome_len) {
        if(recv_fittest(fitness,
                        genome,
                        genome_len)) {

            send_fittest(*fitness,
                         genome,
                         genome_len);

            return true;
        } else {
            return false;
        }
    }

    void Master::send_fittest(float fitness,
                              unsigned char *genome,
                              int genome_len) {
        if(send_requests) {
            // These are all complete now. We just need to clean up the MPI resources.
            for(int i = 0; i < world_size; i++) {
                MPI_Wait(send_requests + i, MPI_STATUS_IGNORE);
            }
        } else {
            send_requests = new MPI_Request[world_size];
        }

        // We only need to create one send buffer, which we send to everyone.
        Message_Fittest::create(&send_buffer,
                                &send_buffer_len,
                                genome_len,
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

    bool Master::recv_fittest(float *fitness,
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
            int n = pending_recvs_count;
            int i_remaining = 0;
            for(int i = 0; i < n; i++) {
                int complete;
                MPI_Test(recv_requests + i, &complete, MPI_STATUS_IGNORE);
                if(!complete) {
                    recv_requests[i_remaining++] = recv_requests[i];
                } else {
                    pending_recvs_count--;
                }
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

                *fitness = max_fitness;
                memcpy(genome,
                       Message_Fittest::get_genome(recv_buffers[i_max_fitness]),
                       genome_len);
            }
        }

        if(pending_recvs_count == 0) {
            for(int i = 0; i < world_size; i++) {
                MPI_Irecv(recv_buffers[i],
                          recv_buffer_lens[i],
                          MPI_CHAR,
                          0,
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
    require( gpu_index > -1 );
    require( 0 == sem_post(sem) );

    require( cudaSuccess == cudaSetDevice(shared_memory->gpu_state[gpu_index].cuda_index) );
}
