/*
  bfs mpi 
  compile: make
  run: mpirun -n <nprocs> bfs <no_of_nodes> <start_node> <end_node> <percent>
 */

#include <mpi.h>
#include <cmath>
#include <math.h>
#include <climits>
#include <vector>
#include <array>
#include <iostream>
#include <sys/time.h>
#include <random>

#include "common_functions_bfs.h"

using namespace std;
using std::cout;
using std::cerr;
using std::endl;

#define MASTER 0
#define ROOT 0
/*
  nVVBuf(local) - 0 for no update, 1 for update
  checkBuf(global) - shared nVV
  srcPtrs, dst, srcPtrs_size, dst_size -> graph
  levelBuf - iteration level for vertex
 */
#define MAX_BUF_SIZE 1<<25
int levelBuf[MAX_BUF_SIZE], nVVBuf[MAX_BUF_SIZE], checkBuf[MAX_BUF_SIZE];



void output_vec(int * data, int data_size, int my_rank) {
  printf("my_rank: %d arr: ", my_rank);
  for (int i = 0; i < data_size; i++)
   printf("%d ", data[i]);
  printf("\n");
}
class CSR {
   public:
      int* srcPtrs;
      int srcPtrs_size;
      int* dst;
      int dst_size;
  CSR(int no_of_nodes, float percent) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0,1/percent);
    vector<int> v_srcPtrs;
    vector<int> v_dst;
    int row, col, edge;
    for (int i = 0; i < no_of_nodes * no_of_nodes; i++) {
      row = i / no_of_nodes;
      col = i % no_of_nodes;

      edge = dis(gen) < 1 ? 1 : 0;

      if (edge != 0) {

	if (row == v_srcPtrs.size()) {
	  v_srcPtrs.push_back(v_dst.size());
	}
	v_dst.push_back(col);
      }
      if (col == no_of_nodes - 1 && row == v_srcPtrs.size())
	v_srcPtrs.push_back(v_dst.size());
    }
    v_srcPtrs.push_back(v_dst.size());
    srcPtrs_size = v_srcPtrs.size();
    dst_size = v_dst.size();
    srcPtrs = new int[srcPtrs_size];
    dst = new int[dst_size];
    for (int i = 0; i < srcPtrs_size; i++ ) {
      srcPtrs[i] = v_srcPtrs[i];
    }  
    for (int i = 0; i < dst_size; i++) {
      dst[i] = v_dst[i];
    }
    
  }
      CSR(int * graph, int no_of_nodes){
      	  vector<int> v_srcPtrs;
	  vector<int> v_dst;
	  v_srcPtrs.push_back(0);
	  for (int i = 0; i < no_of_nodes*no_of_nodes; i ++) {

      	      int row = i / no_of_nodes;
	      int col = i % no_of_nodes;
	      if (graph[i] != 0){
		 // new row
		 if (row == v_srcPtrs.size()) {
		    v_srcPtrs.push_back(v_dst.size());
		 }
		 v_dst.push_back(col);
	      }
	      if (col == no_of_nodes - 1 && row == v_srcPtrs.size())
	      	 v_srcPtrs.push_back(v_dst.size());
	  }
	  v_srcPtrs.push_back(v_dst.size());
	  srcPtrs_size = v_srcPtrs.size();
	  srcPtrs = new int[srcPtrs_size];
	  dst_size = v_dst.size();
	  dst = new int[dst_size];
	  for (int i = 0; i < srcPtrs_size; i++){
	      srcPtrs[i] = v_srcPtrs[i];
	  }
	  for (int i = 0; i < dst_size; i ++){
	      dst[i] = v_dst[i];
	  }
      }
};
bool checkNVV(int nprocs) {
  for (int i = 0; i < nprocs; i++) {
    if (checkBuf[i] == 1)
      return true;
  }
  return false;
    
}
  
void aggregate(int * buffer_level_recv, int my_rank,  int my_work, int nprocs, int * level) {
  for (int i = 0; i < my_work; i++) {
    for (int j = 0; j < nprocs; j++ ) {
      if (buffer_level_recv[i+j*my_work] < level[i+my_rank*my_work]) {
	level[i+my_rank*my_work] = buffer_level_recv[i+j*my_work];

	nVVBuf[my_rank] = 1;
	//	printf("my_rank: %d levelBuf: %d bufflevelrecv: %d nVV: %d\n", my_rank, i+my_rank*my_work, j*my_work, nVVBuf[my_rank]);
      }
    }
  }
}

void bfs(int * srcPtrs, int * dst, int * level, int * nVV, int my_rank, int my_work, int currLevel) {

  // printf("rank: %d my_work: %d check frontier\n",my_rank, my_work);

    for (int i = 0; i < my_work; i ++) {
      // printf("my_rank: %d\n", my_rank);
      if (level[i+my_rank*my_work] == currLevel -1) {
	int start = srcPtrs[my_rank*my_work+i];
	int end = srcPtrs[my_rank*my_work+i+1];
	// printf("my_rank: %d start: %d end: %d\n",my_rank, start, end);
	for (int j = start; j < end; j++ ) {
	  int neighbor = dst[j];
	  // printf("my_rank: %d neighbor: %d, level: %d\n",my_rank, neighbor, levelBuf[neighbor]);
	  if (level[neighbor] == INT_MAX) {
	    // printf("my_rank: %d neighbor: %d, level: %d\n",my_rank, neighbor, levelBuf[neighbor]);
	    level[neighbor] = currLevel;
	    nVV[my_rank] = 1;
	    
	    // printf("my_rank: %d, neighbor: %d, level: %d, nVV: %d\n",my_rank, neighbor, levelBuf[neighbor], nVVBuf[my_rank]);
	  }
	}
      }
    }
}



int main(int argc, char * argv[]) {
  int my_work, my_rank, nprocs;
  int no_of_nodes;

  CSR * csr;

  int start_node, end_node;
  float percent;
  int * srcPtrs;
  int * dst;
  int *srcPtrs_size_arr =  new int[1];
  int *dst_size_arr = new int[1];
  int srcPtrs_size;
  int dst_size;

  long bfs_start, bfs_end, bfs_elapsed;

  struct timeval timecheck;
  
  int * nVV;
  int * level;

  int bfs_result = -1;
  
  MPI_Comm world = MPI_COMM_WORLD;
  //printf("before mpi_init\n");
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  // Initialization
  if (argc == 5) {
    int i = 1;
    no_of_nodes = atoi(argv[i++]);
    start_node = atoi(argv[i++]);
    end_node = atoi(argv[i++]);
    percent = atof(argv[i++]);
  } else {
    no_of_nodes = 10;
    start_node = 0;
    end_node = no_of_nodes -1;
    percent = 0.3;
  }
  if (start_node >= no_of_nodes || end_node >= no_of_nodes || start_node < 0 || end_node < 0) {
    printf("Error: start_node %d or end_node %d has to be valid node[0-%d]\n", start_node, end_node, no_of_nodes-1);
  }
  //printf("my_rank: %d, argument initialized\n", my_rank);
  level = new int[no_of_nodes];
  nVV = new int[nprocs];
  // create level
  for (int i = 0; i < no_of_nodes; i++)
    level[i] = INT_MAX;
  level[start_node] = 0;
  for (int i = 0; i < nprocs; i++)
    nVV[i] = 1;
  printf("my_rank: %d start_node: %d, end_node: %d\n", my_rank, start_node, end_node);
  if (my_rank == ROOT) {
    // Create graph
    csr = new CSR(no_of_nodes, percent);
    srcPtrs_size = csr -> srcPtrs_size;
    dst_size = csr -> dst_size;
    srcPtrs_size_arr[0] = srcPtrs_size;
    dst_size_arr[0] = dst_size;
    srcPtrs = new int[srcPtrs_size];
    dst = new int[dst_size];
    srcPtrs = csr -> srcPtrs;
    dst = csr -> dst;
    
    // output_vec(srcPtrs,srcPtrs_size, my_rank);
    //output_vec(dst,dst_size, my_rank);
      // broadcast graph

    printf("my_rank: %d nprocs: %d no_of_nodes: %d edge: %d\n", my_rank, nprocs, no_of_nodes, dst_size);
    
  }
  // broadcast graph
  //printf("my_rank: %d before broadcast graph\n", my_rank);
  MPI_Bcast(srcPtrs_size_arr, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(dst_size_arr, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("my_rank: %d srcPtrs_size: %d dst_size: %d\n", my_rank, srcPtrs_size, dst_size);
  srcPtrs_size = srcPtrs_size_arr[0];
  dst_size = dst_size_arr[0];
  if (my_rank != ROOT) {
    srcPtrs = new int[srcPtrs_size];
    dst = new int[dst_size];
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(srcPtrs, srcPtrs_size, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(dst, dst_size, MPI_INT, ROOT, MPI_COMM_WORLD);
  

  MPI_Barrier(MPI_COMM_WORLD);
  my_work = no_of_nodes/nprocs;
  if (no_of_nodes%nprocs != 0)
    my_work++;



  printf("my_rank: %d after bcast graph nprocs: %d no_of_nodes: %d edge: %d\n", my_rank, nprocs, no_of_nodes, dst_size);
  
  


  gettimeofday(&timecheck, NULL);
  bfs_start = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;
  int currLevel = 1;


  int compare = 1;

  int * buffer_level_recv = new int[my_work*nprocs];
  MPI_Allgather(&nVV[my_rank], 1, MPI_INT, checkBuf, 1, MPI_INT, MPI_COMM_WORLD);
  //output_vec(nVV, nprocs, my_rank);
  while (checkNVV(nprocs)) {
    // reset flags
 
    nVV[my_rank] = 0;

    //printf("inside while, my_rank: %d\n", my_rank);

    bfs_cuda(srcPtrs, dst, level, nVV, my_rank, my_work,  currLevel, srcPtrs_size, dst_size, no_of_nodes, nprocs);
    currLevel ++;
    
    // output_vec(levelBuf,no_of_nodes, my_rank);
    //printf("my_rank: %d, condition: %d\n", my_rank, nVVBuf[my_rank]);
    // compare -- all processes are in loop
    // output_vec(checkBuf, nprocs, my_rank);

    MPI_Allgather(&nVV[my_rank], 1, MPI_INT, checkBuf, 1, MPI_INT, MPI_COMM_WORLD);
    // output_vec(checkBuf, nprocs, my_rank);
    if (currLevel%compare == 0) {
      //printf("my_rank: %d compare\n", my_rank);

      MPI_Alltoall(level, my_work, MPI_INT, buffer_level_recv, my_work, MPI_INT, world);
      //printf("my_rank: %d after mpi alltoall\n", my_rank);
      //output_vec(buffer_level_recv, my_work*nprocs, my_rank);

      aggregate(buffer_level_recv, my_rank, my_work,nprocs, level);
      //printf("my_rank: %d after mpi aggregate\n", my_rank);      
      // output_vec(level, no_of_nodes, my_rank);
      //printf("my_rank: %d nVV flag: %d\n", my_rank, checkNVV(nprocs));

      MPI_Allgather(&nVV[my_rank], 1, MPI_INT, checkBuf, 1, MPI_INT, MPI_COMM_WORLD);
      //printf("my_rank: %d after all gather nVV flag: %d\n", my_rank, checkNVV(nprocs));
      }
  }

  //  output_vec(level, no_of_nodes, my_rank);
  if (my_rank == end_node/my_work) {
    if (level[end_node] != INT_MAX)
      printf("bfs_result: path exists\n");
    else
      printf("bfs_result: path dne\n");
    gettimeofday(&timecheck, NULL);  
    bfs_end = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;
    bfs_elapsed = bfs_end - bfs_start;


    printf("***********************\n");
    printf("nodes: %d nprocs: %d time: %ld msecs\n", no_of_nodes, nprocs, bfs_elapsed);
  }
  MPI_Finalize();




  return 0;

}
