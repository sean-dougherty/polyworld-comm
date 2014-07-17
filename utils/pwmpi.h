#pragma once

#include <mpi.h>

namespace pwmpi {
    void init(int *argc, char ***argv);
    void finalize();

    bool is_master();
    int rank();
    int size();

    void bld_lock();
    void bld_unlock();

    void gpu_lock();
    void gpu_unlock();

    class Worker {
    public:
        void send_fittest(int generation,
                          long agent_id,
                          float fitness,
                          unsigned char *genome,
                          int genome_len);

        bool recv_fittest(int generation,
                          long &agent_id,
                          float &fitness,
                          unsigned char *genome,
                          int genome_len);
    private:
        int last_generation_sent = 0;
        MPI_Request send_request = MPI_REQUEST_NULL;
        unsigned char *send_buffer = nullptr;
        int send_buffer_len = 0;

        int last_generation_received = 0;
        MPI_Request recv_request = MPI_REQUEST_NULL;
        unsigned char *recv_buffer = nullptr;
        int recv_buffer_len = 0;
    };
    extern Worker *worker;

    class Master {
    public:
        void update_fittest(int genome_len);

    private:
        void send_fittest(long agent_id,
                          float fitness,
                          unsigned char *genome,
                          int genome_len);

        bool recv_fittest(long &agent_id,
                          float &fitness,
                          unsigned char *genome,
                          int genome_len);

    private:
        MPI_Request *send_requests = nullptr;
        unsigned char *send_buffer = nullptr;
        int send_buffer_len = 0;

        MPI_Request *recv_requests = nullptr;
        int pending_recvs_count = 0;
        unsigned char **recv_buffers = nullptr;
        int *recv_buffer_lens = 0;
    };
    extern Master *master;
}
