/*
	bfs cuda mpi
	compile: make
	run: mpirun -n <nprocs> bfs <number_of_nodes> <start_node> <end_node> <percent>
*/
#include <iostream>
#include <cmath>
#include <math.h>
#include <vector>
#include <limits.h>
#include <random>

using namespace std;

using std::cout;
using std::cerr;
using std::endl;

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "common_functions.h"
#define NO_OF_NODES 9;

void out_vec(int * arr, int arr_size, int my_rank) {
     printf("my_rank: %d arr: ", my_rank);
     for (int i = 0; i < arr_size; i++){
     	 printf("%d ", arr[i]);
     }
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
    for (int i = 0; i < srcPtrs_size; i++ )
      srcPtrs[i] = v_srcPtrs[i];
    for (int i = 0; i < dst_size; i++)
      dst[i] = v_dst[i];
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


__global__ void bfs_kernel(int * d_srcPtrs, int* d_dst, int my_rank, int my_work, int * level, int currLevel, int * newVertexVisited){
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	if (i < my_work) {
	   int index = my_rank*my_work+i;
	   if (level[index] == currLevel -1){

	      int start = d_srcPtrs[index];
	      int end = d_srcPtrs[index+1];

	      for(int j = start; j < end; j++) {

	      	      int neighbor = d_dst[j];

		      if (level[neighbor] == INT_MAX) {

		      	 level[neighbor] = currLevel;
			 newVertexVisited[my_rank] = 1;


		      }
	      }
	   }

	}
}

int bfs_cuda(int* h_srcPtrs, int*h_dst, int * h_level, int * h_nVV, int my_rank, int my_work, int currLevel, int srcPtrs_size, int dst_size, int no_of_nodes, int nprocs)
{

   //  printf("my_rank: %d, inside bfs_cuda currLevel: %d\n", my_rank, currLevel);

    int* d_srcPtrs,* d_dst,*d_level, *d_nVV = 0;

    unsigned int srcPtrs_bytes = srcPtrs_size*sizeof(int);

//    printf("my_rank: %d before cudaMalloc srcPtrs_size: %d dst_size: %d, no_of_nodes: %d nprocs: %d\n", my_rank, srcPtrs_size, dst_size, no_of_nodes, nprocs);
//    out_vec(h_srcPtrs, srcPtrs_size, my_rank);
//    out_vec(h_dst, dst_size, my_rank);
//    out_vec(h_level, no_of_nodes, my_rank);
//    out_vec(h_nVV, nprocs, my_rank);
//    printf("my_rank: %d cudaMalloc start cudaEvent start\n", my_rank);
//    printf("my_rank: %d cudaMalloc start\n", my_rank);
    cudaMalloc(reinterpret_cast<void **>(&d_srcPtrs), srcPtrs_bytes);
//       printf("my_rank: %d cudaMalloc after first\n", my_rank);
    cudaMalloc(reinterpret_cast<void **>(&d_dst), dst_size*sizeof(int));
//        printf("my_rank: %d cudaMalloc after second\n", my_rank);
    cudaMalloc(reinterpret_cast<void **>(&d_level), no_of_nodes*sizeof(int));
//        printf("my_rank: %d cudaMalloc after third\n", my_rank);
    cudaMalloc(reinterpret_cast<void **>(&d_nVV), nprocs*sizeof(int));

//   printf("my_rank: %d before cudaMemcpy after cudaMalloc\n", my_rank);
    cudaMemcpy(d_srcPtrs, h_srcPtrs, srcPtrs_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, dst_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, h_level, no_of_nodes*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_nVV, h_nVV, nprocs*sizeof(int), cudaMemcpyHostToDevice);

    int threads = 100;
    int grid = my_work/threads;
    if (my_work%threads != 0)
       grid++;

  //  printf("my_rank: %d enter kernel function\n", my_rank);
       
    bfs_kernel<<<grid, threads>>>(d_srcPtrs, d_dst, my_rank, my_work, d_level, currLevel, d_nVV);

    cudaMemcpy(h_nVV, d_nVV, nprocs*sizeof(int), cudaMemcpyDeviceToHost);	  
    cudaMemcpy(h_level, d_level, no_of_nodes*sizeof(int), cudaMemcpyDeviceToHost);
 //   out_vec(h_level, no_of_nodes, my_rank);
    return 0;

}
/*
int main(int argc, char* argv[]){
    
    int no_of_nodes;
    int start_node;
    int end_node;
    float percent;
    CSR * csr;

    long bfs_start, bfs_end, bfs_elapsed;
      cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);

  
  printf("prop.name=%s\n", prop.name);
  printf("prop.multiProcessorCount=%d\n", prop.multiProcessorCount);
  printf("prop.major=%d minor=%d\n", prop.major, prop.minor);
  printf("prop.maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
  printf("maxThreadsDim.x=%d maxThreadsDim.y=%d maxThreadsDim.z=%d\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
  printf("prop.maxGridSize.x=%d maxGridSize.y=%d maxGridSize.z=%d\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  printf("prop.maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
  printf("prop.totalGlobalMem=%u\n", prop.totalGlobalMem);
  printf("prop.regsPerBlock=%d\n", prop.regsPerBlock);
  
  printf("\n");


    struct timeval timecheck;
    if (argc == 5){
       int i = 1;
       no_of_nodes = atoi(argv[i++]);
       start_node = atoi(argv[i++]);
       end_node = atoi(argv[i++]);
       percent = atof(argv[i++]);
    } else {
      no_of_nodes = 10;
      start_node = 0;
      end_node = no_of_nodes -1;
      percent = 0.2;
    }
    if (start_node >= no_of_nodes || end_node >= no_of_nodes || start_node < 0 || end_node < 0) {
    printf("Error: start_node %d or end_node %d has to be valid node[0-%d]\n", start_node, end_node, no_of_nodes);
      }  
      
    csr = new CSR(no_of_nodes, percent);

    int * srcPtrs = csr -> srcPtrs;
    int * dst = csr -> dst;
    int srcPtrs_size = csr -> srcPtrs_size;
    int dst_size = csr -> dst_size;
    
    int * level = new int[no_of_nodes];
    for (int i = 0; i < no_of_nodes; i++)
    	level[i] = INT_MAX;
    level[start_node] = 0;

    int nprocs = 2;
    int my_rank = 0;
    int currLevel = 1;
    int my_work = no_of_nodes/nprocs;
    if (no_of_nodes%nprocs != 0)
       my_work ++;
    int * nVV = new int[nprocs];
    for (int i = 0; i < nprocs; i++)
    	nVV[i] = 0;
    nVV[my_rank] = 1;
    int o_dst_size = csr -> dst_size;
   
    printf("dst_size: %d\n", o_dst_size);
    printf("start: %d\n", start_node);
    printf("end: %d\n", end_node);

    out_vec(srcPtrs, srcPtrs_size, my_rank);
    out_vec(dst, dst_size, my_rank);
    
    int grid = 1000;
    int threads = no_of_nodes/grid;
    if (no_of_nodes%grid != 0)
       threads++;

    gettimeofday(&timecheck, NULL);
    bfs_start = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec/ 1000;
    while (nVV[my_rank] == 1) {
    	  nVV[my_rank] = 0;
    	  int bfs_result = bfs_cuda(srcPtrs,dst,level,nVV,my_rank, my_work, currLevel, srcPtrs_size, dst_size,  no_of_nodes,nprocs);
	  currLevel ++;
    }
    if (level[end_node] != INT_MAX)
      printf("bfs_result: path exists\n");
    else
      printf("bfs_result: path dne\n");

    gettimeofday(&timecheck, NULL);
    bfs_end = (long) timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    bfs_elapsed = bfs_end - bfs_start;

    printf("*******************\n");
    printf("nodes: %d grid: %d threads: %d time: %ld msecs\n", no_of_nodes, grid, threads, bfs_elapsed);
}*/