#pragma once

namespace pwmpi {
    void init(int *argc, char ***argv);
    void finalize();

    bool is_master();

    int get_gpu_index();
    void gpu_lock();
    void gpu_unlock();
}
