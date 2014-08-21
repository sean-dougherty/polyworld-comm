#pragma once

#ifdef INCLUDE_MPI
#include <mpi.h>
#else
typedef void *MPI_Request;
#endif

namespace pwmpi {
    void init(int *argc, char ***argv);
    void finalize();

    bool is_master();
    int rank();
    int size();

    int get_demes_count(int max_demes);

    void gpu_lock();
    void gpu_unlock();

    class Worker {
    public:
        Worker();

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

        bool is_simulation_ended();
    private:
        int last_generation_sent;
        MPI_Request send_request;
        unsigned char *send_buffer;
        int send_buffer_len;

        int last_generation_received;
        MPI_Request recv_request;
        unsigned char *recv_buffer;
        int recv_buffer_len;
    };
    extern Worker *worker;

    class Master {
    public:
        Master();

        void update_fittest(int genome_len);
        void end_simulation();

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
        MPI_Request *send_requests;
        unsigned char *send_buffer;
        int send_buffer_len;

        MPI_Request *recv_requests;
        int pending_recvs_count;
        unsigned char **recv_buffers;
        int *recv_buffer_lens;
    };
    extern Master *master;
}
