/*
  BFS Serial
  compile : g++ -o bfs bfs.cpp
  run: ./bfs
  ./bfs <no_of_nodes> <start_node> <end_node>
 */
#include <queue>
#include <stack>
#include <set>
#include <cmath>
#include <math.h>

#include <iostream>
#include <unordered_set>
#include <sys/time.h>
#include <random>

using namespace std;
using std::cout;
using std::cerr;
using std::endl;

#define NO_OF_NODES 6;

void init_graph(int* graph,int no_of_nodes, float percent);
int bfs(int* graph, int start_node, int end_node, int no_of_nodes);
void output_vec(int * data, int data_size) {
  for (int i = 0; i< data_size; i++)
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
      // printf("row: %d col: %d, edge: %d\n", row, col, edge);
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
//	      printf("index: %d graph: %d srcPtrs_size: %d dst_size %d\n", i, graph[i], v_srcPtrs.size(), v_dst.size());
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


// For now, only return path existence
int bfs(CSR * csr,int start_node,int end_node, int no_of_nodes){
  int * srcPtrs = csr -> srcPtrs;
  int * dst = csr-> dst;
  int srcPtrs_size = csr -> srcPtrs_size;
  int dst_size = csr -> dst_size;
  // printf("edges: %d\n", dst_size);
  //output_vec(srcPtrs,srcPtrs_size);
  //output_vec(dst, dst_size);
  queue<int> neighbors;
  set<int> visited;
  neighbors.push(start_node);
  while (!neighbors.empty()){
    int node = neighbors.front();

    // end node found
    if (node == end_node)
      return 1;
    // node is not in visited
    if (auto iter = visited.find(node); iter == visited.end()){
      int start = srcPtrs[node];
      int end = srcPtrs[node+1];

      for (int j = start; j < end; j ++) {
	  neighbors.push(dst[j]);
      }
    }
    // printf("Size of queue: %d \n",neighbors.size());
    visited.insert(node);
    neighbors.pop();    

  }
  // end could not be reached
  return -1;
}
// adjacency matrix
void init_graph(int* graph, int no_of_nodes, float percent = 0.05) {
      // directed, unweighted
      random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0,1/percent);
  for (int i = 0 ; i < no_of_nodes*no_of_nodes; i++ ) {
    graph[i] = dis(gen) < 1 ? 1 : 0;
  }
  }

void output_graph(int * graph, int no_of_nodes){
  for (int i = 0; i < no_of_nodes*no_of_nodes; i ++){
    if (i%no_of_nodes == 0){
      printf("\n");
    }
    printf("%*d ",3, graph[i]);

  }
 printf("\n");  
}
// single source to single destination path
// undirected and weighted graph
int main(int argc, char* argv[]) {
  int * graph;
  int no_of_nodes;
  int start_node, end_node;
  float percent;
  CSR * csr;

  long bfs_start, bfs_end, bfs_elapsed;


  struct timeval timecheck;
  if (argc == 5) {
    int i = 1;
    no_of_nodes = atoi(argv[i++]);
    start_node = atoi(argv[i++]);
    end_node = atoi(argv[i++]);
    percent = atof(argv[i++]);
  }
  else {
    no_of_nodes = 10;
    start_node = 0;
    end_node = no_of_nodes -1;
    percent = 0.2;
  }
  if (start_node >= no_of_nodes || end_node >= no_of_nodes || start_node < 0 || end_node < 0) {
    printf("Error: start_node %d or end_node %d has to be valid node[0-%d]\n", start_node, end_node, no_of_nodes);
  }  
  //  graph = new int[no_of_nodes*no_of_nodes];
  //  init_graph(graph, no_of_nodes, percent);
  csr = new CSR(no_of_nodes,percent);
  //output_graph(graph, no_of_nodes);
  printf("start : %d\n",start_node);
  printf("end : %d\n", end_node);
  
  gettimeofday(&timecheck, NULL);
  bfs_start = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;
  int bfs_result = bfs(csr,start_node, end_node, no_of_nodes);
  if (bfs_result == 1)
    printf("bfs_result : path exists\n");
  if (bfs_result == -1)
    printf("bfs_result: path dne\n");
  gettimeofday(&timecheck, NULL);  
  bfs_end = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;
  bfs_elapsed = bfs_end - bfs_start;


  printf("***********************\n");
  printf("nodes: %d time: %ld msecs\n", no_of_nodes, bfs_elapsed);
    
  return 0;
}
