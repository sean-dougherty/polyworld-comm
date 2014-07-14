#pragma once

namespace pwmpi {
    void init(int *argc, char ***argv);
    void finalize();

    bool is_mpi_mode();
    bool is_master();
    int rank();

    void bld_lock();
    void bld_unlock();

    int get_gpu_index();
    void gpu_lock();
    void gpu_unlock();
}
