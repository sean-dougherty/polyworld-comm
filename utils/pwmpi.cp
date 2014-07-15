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
            if(*buffer) {
                return;
            }
            int message_size = get_message_size(genome_len);
            *buffer_len = message_size;
            *buffer = (unsigned char *)malloc(message_size);
        }

        static void create(unsigned char **buffer,
                           int *buffer_len,
                           int genome_len,
                           float fitness,
                           unsigned char *genome) {
            alloc(buffer, buffer_len, genome_len);

            Message_Fittest *header = (Message_Fittest *)*buffer;
            header->fitness = fitness;
            header->genome_len = genome_len;
        
            unsigned char *payload = (unsigned char *)(header + 1);
            memcpy(payload, genome, genome_len);
        }
    };


    void Worker::send_fittest(int generation,
                              float fitness,
                              unsigned char *genome,
                              int genome_len) {

        if(send_request != MPI_REQUEST_NULL) {
            int complete;
            MPI_Test(&send_request, &complete, MPI_STATUS_IGNORE);
            if(!complete)
                return;
        }

        if( last_generation_received < last_generation_sent )
            return;

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

        if(recv_request != MPI_REQUEST_NULL) {
            int complete;
            MPI_Test(&recv_request, &complete, MPI_STATUS_IGNORE);
            if(!complete)
                return false;

            received = true;
            last_generation_received = generation;

            Message_Fittest *header = (Message_Fittest *)recv_buffer;
            require(header->genome_len == genome_len);
            *fitness = header->fitness;
            unsigned char *payload = (unsigned char *)(header + 1);
            memcpy(genome, payload, genome_len);
        }

        Message_Fittest::alloc(&recv_buffer,
                               &recv_buffer_len,
                               genome_len);

        MPI_Irecv(recv_buffer,
                  recv_buffer_len,
                  MPI_CHAR,
                  0,
                  Tag_Master_Send_Fittest,
                  MPI_COMM_WORLD,
                  &recv_request);

        return received;
    }

    void Master::send_fittest(float fitness,
                              unsigned char *genome,
                              int genome_len) {
        if(send_requests) {
            for(int i = 0; i < world_size; i++) {
                MPI_Wait(send_requests + i, MPI_STATUS_IGNORE);
            }
        } else {
            send_requests = new MPI_Request[world_size];
        }

        Message_Fittest::alloc(&send_buffer,
                               &send_buffer_len,
                               genome_len);

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
