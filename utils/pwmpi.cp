#include "pwmpi.h"

#include "pwassert.h"

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

static bool mpi_mode = false;

static sem_t *sem = nullptr;
static int gpu_index = -1;
static int world_rank;
static int world_size;

const int Tag_Init = 0;
const int Tag_Node_Ranks = 1;

#define MAX_NODE_RANKS 1024
#define MAX_GPUS 5

struct shared_memory_t {
    bool initialized;
    size_t nranks;
    int ranks[MAX_NODE_RANKS];

    size_t ngpus;
    struct gpu_state_t {
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

        shared_memory->ngpus = 1;
        for(size_t i = 0; i < shared_memory->ngpus; i++) {
            shared_memory_t::gpu_state_t &gpu = shared_memory->gpu_state[i];

            require( 0 == sem_init(&gpu.sem, 1, 1) );

            gpu.process_slots_available = 2;
        }
    }

    if(!is_master()) {
        require( (shared_memory->nranks + 1) < MAX_NODE_RANKS );
        shared_memory->ranks[shared_memory->nranks++] = world_rank;

        for(size_t i = 0; i < shared_memory->ngpus; i++) {
            shared_memory_t::gpu_state_t &gpu = shared_memory->gpu_state[i];
        
            if(gpu.process_slots_available > 0) {
                gpu.process_slots_available--;
                gpu_index = (int)i;
                break;
            }
        }
        require( gpu_index > -1 );
    }

    require( 0 == sem_post(sem) );

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

int get_gpu_index() {
    return gpu_index;
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

}
