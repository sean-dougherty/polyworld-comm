#include "pwmpi.h"

#include "pwassert.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>

static sem_t *gpu_sem = nullptr;

namespace pwmpi {

void init() {
    require( gpu_sem = sem_open("/pwmpi-gpu",
                                O_CREAT,
                                S_IRWXU,
                                1) );
}

void gpu_lock() {
    assert(gpu_sem);

    printf("waiting on gpu sempahore..."); fflush(stdout);
    require( 0 == sem_wait(gpu_sem) );
    printf(" OK!\n"); fflush(stdout);
}

void gpu_unlock() {
    assert(gpu_sem);

    printf("posting gpu sempahore..."); fflush(stdout);
    require( 0 == sem_post(gpu_sem) );
    printf(" OK!\n"); fflush(stdout);
}

}
