/**
 * @author RookieHPC
 * @brief Original source code at https://rookiehpc.org/mpi/docs/mpi_bcast/index.html
 **/

#include <stdio.h>
#include <iostream>
#include <mpi.h>

/**
 * @brief Illustrates how to broadcast a message.
 * @details This code picks a process as the broadcast root, and makes it
 * broadcast a specific value. Other processes participate to the broadcast as
 * receivers. These processes then print the value they received via the 
 * broadcast.
 **/
#define MAX_BUF_SIZE 1<<25
#define ROOT 0
int nVVBuf[MAX_BUF_SIZE];
void output_vec(int * data, int data_size , int my_rank){
  printf("my_rank %d: ", my_rank);
  for (int i = 0; i < data_size; i++) {
    printf("%d ", data[i]);
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // Get my rank in the communicator
    int my_rank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Determine the rank of the broadcast emitter process
    int broadcast_root = 0;

    int buffer;
    nVVBuf[my_rank] = 1;
    int * isRunning = new int[nprocs];
    MPI_Barrier(MPI_COMM_WORLD);
      output_vec(nVVBuf, nprocs, my_rank);
      MPI_Allgather(&nVVBuf[my_rank], 1, MPI_INT, isRunning,1, MPI_INT, MPI_COMM_WORLD);
      //    MPI_Bcast(nVVBuf+my_rank, 1, MPI_INT, my_rank, MPI_COMM_WORLD);
    output_vec(isRunning, nprocs, my_rank);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
