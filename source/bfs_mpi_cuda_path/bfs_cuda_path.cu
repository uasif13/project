/*
	astar cuda no mpi
	compile: nvcc -ICommon -I/usr/local/include/igraph -L/usr/local/lib -ligraph astar_cuda.cu
	run: ./a.out < grid_size> <start> <end> <percent> <obstacle_type> <heuristic>
*/
#include <igraph.h>
#include <stdio.h>
#include <cmath>
#include <math.h>

#include <iostream>
#include <sys/time.h>
#include <random>
#include <climits>
#include <cstdint>
#include <inttypes.h>

using namespace std;
using std::cout;
using std::cerr;
using std::endl;

#include "common_functions_bfs_path.h"
/*extern "C"{
   int astar_cuda(uint64_t * openlist, uint64_t * closedList, uint64_t * src, uint64_t * dst, uint64_t * gList, uint64_t * hList, uint64_t * parent, uint64_t * nVV, int my_rank, uint64_t my_work, uint64_t srcPtrs_size, uint64_t dst_size, uint64_t grid_size, uint64_t no_of_nodes, int nprocs, uint64_t heuristic_type, uint64_t end, int iteration);
}*/

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

int astar(uint64_t* src, uint64_t* dst, uint64_t no_of_nodes, uint64_t start, uint64_t end, string heuristic);


void output_vec(int * data, int data_size) {
  for (int i = 0; i < data_size; i ++ ) {
    printf("%d ", data[i]);
  }
  printf("\n");
}

void output_graph(int * graph, int graph_size) {
  for (int i = 0; i < graph_size*graph_size; i++) {
    if (i%graph_size == 0)
      printf("\n");
    printf("%*d", 7, graph[i]);
    
  }
  printf("\n");
}

void output_tail(int* graph, int graph_size, int tail_count) {
     uint64_t no_of_nodes = graph_size*graph_size;
     for (int i = 5 ; i >= 1; i--) {
     	 printf("%*" PRIu64  "", 3, graph[no_of_nodes-i]);
     }
     printf("\n");
}
void output_score(uint64_t * data, uint64_t grid_size) {
     for (uint64_t i = 0; i < grid_size*grid_size; i++) {
     	 if (i%grid_size == 0)
	    printf("\n");
     	 printf("%*" PRIu64  "", 10, data[i]);
     }
     printf("\n");
}

void output_end(uint64_t * data, uint64_t data_size, uint64_t tail_count) {
     for (uint64_t i = tail_count; i >= 1; i--) {
     	 printf("%*" PRIu64  "", 10, data[data_size-i]);
     }
     printf("\n");
}
void output_head(uint64_t * data, uint64_t data_size, uint64_t head_count) {
     for (uint64_t i = 0; i < head_count; i++) {
     	 printf("%*" PRIu64  "", 15, data[i]);
     }
     printf("\n");
}



__global__ void calculate_heuristic(uint64_t * d_hList, uint64_t grid_size, uint64_t no_of_nodes, uint64_t end, uint64_t heuristic_type) {
	   uint64_t i = blockDim.x*blockIdx.x+threadIdx.x;
	   uint64_t row_diff, col_diff;
	   if (i < no_of_nodes) {
	      if (end > i) {
	      	 row_diff = end - i;
		 col_diff = end -i;
	      } else {
	      	row_diff = i - end;
		col_diff = i-end;
	      }
	      row_diff /= grid_size;
	      col_diff %= grid_size;
	      if (heuristic_type == 0)
	      	 d_hList[i] = row_diff + col_diff;
	      else
		d_hList[i] = pow(row_diff*row_diff+col_diff*col_diff, 0.5);
	   }
}

__global__ void astar_k(uint64_t * d_openList, uint64_t * d_closedList, uint64_t * d_src, uint64_t * d_dst, uint64_t * d_gList, uint64_t * d_hList,uint64_t * d_parent, uint64_t * d_flag, uint64_t grid_size, uint64_t no_of_nodes) {
	   uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
	   uint64_t g_score;
	 
	   if (i < no_of_nodes && d_openList[i] == 1) {
	      *d_flag = 1;

	      d_openList[i] = 0;
	      d_closedList[i] = 1;
	      uint64_t start = d_src[i];
	      uint64_t end = d_src[i+1];

	      for (uint64_t j = start; j < end; j ++) {
	      	  
	      	  uint64_t nb = d_dst[j];
		 
		  if (d_closedList[nb] == 1)
		     continue;
		 
		  g_score = d_gList[i]+1;
		  if (d_openList[nb] == 0)
		     d_openList[nb] = 1;
		  else if (d_gList[nb] != 0 && g_score >= d_gList[nb])
		       continue;
		  d_gList[nb] = g_score;
		  d_parent[nb] = i;
		 
		  
	      }
	   }
}
__global__ void astar_kernel(uint64_t * d_openList, uint64_t * d_closedList, uint64_t * d_src, uint64_t * d_dst, uint64_t * d_gList, uint64_t * d_hList,uint64_t * d_parent, uint64_t * d_flag, uint64_t no_of_nodes, int my_rank, int nprocs) {
	   uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
	   uint64_t g_score;
	   uint64_t nb;
	   //if (d_openList[i]== 1)
//	      printf("index: %" PRIu64"\n", i);
//	   if (i == 5001)
//	      printf("my_rank: %d index: %" PRIu64 "\n", my_rank, i);
	   if (i < no_of_nodes && d_openList[i] == 1) {
	      d_flag[my_rank] = 1;

	      d_openList[i] = 0;
	      d_closedList[i] = 1;
	      uint64_t start = d_src[i];
	      uint64_t end = d_src[i+1];
//	      printf("my_rank: %d index: %" PRIu64 " start: %" PRIu64 " end: %" PRIu64"\n", my_rank, i, start, end);
	      for (uint64_t j = start; j < end; j ++) {
	      	 
	      	  nb = d_dst[j];
//		   printf("my_rank: %d after neighbor neighbor: %" PRIu64 " parent: %" PRIu64 " openList: %" PRIu64 "\n", my_rank, nb, i, d_openList[nb]);		 
		 
		  if (d_closedList[nb] == 1)
		     continue;
//		  printf("my_rank: %d after closedList check neighbor: %" PRIu64 " nVVBuf: %" PRIu64 "\n", my_rank, nb, d_flag[my_rank]); 
		  g_score = d_gList[i]+1;
		  if (d_openList[nb] == 0) {
		     d_openList[nb] = 1;
//		     printf("my_rank: %d after openList set neighbor: %" PRIu64 " parent: %" PRIu64 " openList: %" PRIu64 "\n", my_rank, nb, i, d_openList[nb]);		 
		     
		     }
		  else if (d_gList[nb] != 0 && g_score >= d_gList[nb]) {
		       
  //     		     printf("my_rank: %d after g check neighbor: %" PRIu64 " g_score: %" PRIu64 " d_gList: %" PRIu64 "\n", my_rank, nb, g_score, d_gList[nb]);
		     continue;
		       }
		  d_gList[nb] = g_score;
		  d_parent[nb] = i;
//		  printf("my_rank: %d after g set neighbor: %" PRIu64 " g: %" PRIu64 " parent: %" PRIu64 " openList: %" PRIu64 "\n", my_rank, nb, g_score, i, d_openList[nb]);		 
		 
		  
	      }
	   }
}


int astar_cuda (uint64_t * openList, uint64_t* closedList, uint64_t* src, uint64_t* dst, uint64_t* gList, uint64_t* hList, uint64_t* parent, uint64_t* nVVBuf, int my_rank, uint64_t my_work, uint64_t total_src, uint64_t total_dst, uint64_t grid_size, uint64_t no_of_nodes, int nprocs, uint64_t heuristic_type, uint64_t end, int iteration) {

    //printf("inside cuda\n");
    uint64_t * d_hList;
    uint64_t * d_gList;
    uint64_t * d_openList;
    uint64_t * d_closedList;
    uint64_t * d_flag;
    uint64_t * d_src;
    uint64_t * d_dst;
    uint64_t * d_parent;
    

    uint64_t threads = 1000;
    uint64_t grid = no_of_nodes/threads;
    if (no_of_nodes%threads != 0)
       grid++;

  //  printf("my_rank: %d grid: %" PRIu64 " threads: %" PRIu64 "\n", my_rank, grid, threads);

    cudaMalloc(&d_hList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_gList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_openList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_closedList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_flag, nprocs*sizeof(uint64_t));
    cudaMalloc(&d_src, total_src*sizeof(uint64_t));
    cudaMalloc(&d_dst, total_dst*sizeof(uint64_t));
    cudaMalloc(&d_parent, no_of_nodes*sizeof(uint64_t));
    if (iteration == 0) {
//       calculate_heuristic<<<grid, threads>>>(d_hList, grid_size, no_of_nodes, end, heuristic_type);
	cudaMemcpy(hList, d_hList, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpy(d_hList, hList, no_of_nodes*sizeof(uint64_t), cudaMemcpyHostToDevice);
    }


   // output_end(hList, no_of_nodes, 5);

   

    cudaMemcpy(d_openList, openList, no_of_nodes*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_closedList, closedList, no_of_nodes*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src, total_src*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst, total_dst*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gList, gList, no_of_nodes*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, parent, no_of_nodes*sizeof(uint64_t), cudaMemcpyHostToDevice);
    nVVBuf[my_rank] = 0;
    cudaMemcpy(d_flag, nVVBuf, nprocs*sizeof(uint64_t), cudaMemcpyHostToDevice);
    printf("my_rank: %d no_of_nodes: %" PRIu64" \n",my_rank,no_of_nodes);
    astar_kernel<<<grid, threads>>>(d_openList, d_closedList, d_src, d_dst, d_gList, d_hList, d_parent, d_flag, no_of_nodes, my_rank, nprocs);
    cudaMemcpy(nVVBuf, d_flag, nprocs*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
//    output_end(nVVBuf, 2,2);
    cudaMemcpy(openList, d_openList, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(closedList, d_closedList, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(gList, d_gList, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(parent, d_parent, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  //  output_head(gList, no_of_nodes,5);
    //output_score(parent, grid_size);
    return 1;
}

// standalone cuda
int astar (uint64_t* src, uint64_t* dst, uint64_t no_of_nodes, uint64_t grid_size, uint64_t start, uint64_t end, string heuristic, uint64_t * final_path) {
    uint64_t heuristic_type;
    uint64_t no_of_edges = src[no_of_nodes];
    if (heuristic == "manhattan")
       heuristic_type = 0;
    else
	heuristic_type = 1;

   
    uint64_t * hList = new uint64_t[no_of_nodes];

    uint64_t * d_hList;
    uint64_t * d_gList;
    uint64_t * d_openList;
    uint64_t * d_closedList;
    uint64_t * d_flag;
    uint64_t * d_src;
    uint64_t * d_dst;
    uint64_t * d_parent;
    

    uint64_t threads = 1000;
    uint64_t grid = no_of_nodes/threads;
    if (no_of_nodes%threads != 0)
       grid++;

    cudaMalloc(&d_hList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_gList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_openList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_closedList, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_flag, sizeof(uint64_t));
    cudaMalloc(&d_src, no_of_nodes*sizeof(uint64_t));
    cudaMalloc(&d_dst, no_of_edges*sizeof(uint64_t));
    cudaMalloc(&d_parent, no_of_nodes*sizeof(uint64_t));

    calculate_heuristic<<<grid, threads>>>(d_hList, grid_size, no_of_nodes, end, heuristic_type);

    cudaMemcpy(hList, d_hList, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    output_end(hList, no_of_nodes, 5);

    uint64_t * openList = new uint64_t[no_of_nodes];
    uint64_t * closedList = new uint64_t[no_of_nodes];
    uint64_t * gList = new uint64_t[no_of_nodes];
    uint64_t * nVV = new uint64_t[no_of_nodes];
    uint64_t * parent = new uint64_t[no_of_nodes];
    for (uint64_t i = 0; i < no_of_nodes; i++) {
    	openList[i] = 0;
    }

    *nVV = 1;
    openList[start] = 1;

    cudaMemcpy(d_openList, openList, no_of_nodes*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src, no_of_nodes*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst, no_of_edges*sizeof(uint64_t), cudaMemcpyHostToDevice);

    while (*nVV == 1) {
    	  *nVV = 0;
	  cudaMemcpy(d_flag, nVV, sizeof(uint64_t), cudaMemcpyHostToDevice);
	  astar_k<<<grid, threads>>>(d_openList, d_closedList, d_src, d_dst, d_gList, d_hList, d_parent, d_flag, grid_size, no_of_nodes);
	  cudaMemcpy(nVV, d_flag, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(gList, d_gList, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(parent, d_parent, no_of_nodes*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    //output_score(gList, grid_size);
    //output_score(parent, grid_size);
    vector<int> path;
    if (gList[end] != 0) {
       int current = end;
       path.push_back(end);
       while (current != start) {
       	     path.push_back(parent[current]);
	     current = parent[current];
       	     }
       uint64_t path_size = path.size();
   
       for (uint64_t i = 0; i < path_size; i++) {
       	   final_path[i] = path[path_size-i-1];
       }
       return gList[end];
       }
    else
	return -1;
}

// single source to single destination path
