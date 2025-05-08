#include <queue>
#include <stack>
#include <set>
#include <cmath>
#include <math.h>

#include <iostream>
#include <unordered_set>
#include <sys/time.h>

using namespace std;
using std::cout;
using std::cerr;
using std::endl;

#define NO_OF_NODES 6;

void init_graph(int* graph,int no_of_nodes);
int bfs(int* graph, int start_node, int end_node, int no_of_nodes);
int dfs(int* graph, int start_node, int end_node, int no_of_nodes);
int astar(int* graph, int start_node, int end_node, int no_of_nodes);


class NodeAStar {
public:
  int index = 0;
  int g = 0;
  int h = 0;
  int f = g+h;
  NodeAStar(int n_index,int n_g,int n_h){
    index = n_index;
    g = n_g;
    h = n_h;
    f = n_g+n_h;
  }
};
int find_neighbor(vector<NodeAStar*>open ,int open_size, int node_index) {
  for (int i = 0; i < open_size; i++) {
    if (open[i] -> index == node_index){
      printf("neighor %d found at %d\n", node_index,i);
      return i;
    }
  }
  return -1;

}

int find_least_f(vector<NodeAStar*> open, int open_size){
  int least_f_value = open[0] -> f;
  int least_f_index = 0;
  for (int i = 1; i< open_size; i++){
    int f_value = open[i] -> f;
    if (least_f_value > f_value){
      least_f_value = f_value;
      least_f_index = i;
    }
  }
  return least_f_index;
}
// heuristic: math.abs(end_node-node_index)
int astar(int* graph, int start_node, int end_node, int no_of_nodes){
  int start_g = 0;
  int start_h = abs(end_node- start_node);
  NodeAStar * start = new NodeAStar(start_node, start_g, start_h);
  vector<NodeAStar*> open;
  set<int> open_set;
  open.push_back(start);
  open_set.insert(start_node);
  int open_size = 1;
  
  vector<NodeAStar*> closed;
  set<int> closed_set;
  int closed_size = 0;
  
  /*  struct
  {
    bool operator()(const NodeAStar* l, const NodeAStar* r) const{return l->index>r-> index;}
  }customLess;
  set<NodeAStar *,customLess, allocator<NodeAStar*>> closed;
  set <NodeAStar *, customLess, allocator<NodeAStar*>>open;
  auto cmp = [](NodeAStar* left, NodeAStar*right){return left->g < right->g;}; 
  priority_queue<NodeAStar*, vector<NodeAStar*>, decltype(cmp)> open_pq(cmp);
  open_pq.push(start);
  open.push(start);*/

  while (open_size != 0){
    int current_index = find_least_f(open, open_size);
    NodeAStar* node = open[current_index];
    int node_index = node-> index;
    int node_g = node-> g;
    int node_h = node-> h;
    printf("Index: %d, g: %d, h: %d\n",node_index,node_g,node_h);
    if (node_index == end_node){
      return node_g;
    }
    open.erase(open.begin()+current_index);
    open_set.erase(current_index);
    open_size --;
    closed.push_back(node);
    closed_set.insert(node_index);
    closed_size ++;
    printf("open_size: %d, closed_size: %d, no_of_nodes: %d\n",open_size, closed_size, no_of_nodes);    
    int row = node_index*no_of_nodes;
    for (int i = 0; i < no_of_nodes; i++){
      // skip visited nodes
      int check_closed = find_neighbor(closed, closed_size, i);
      printf("i: %d, closed: %d\n",i,check_closed);
      if (i == node_index || check_closed != -1 || graph[row+i] == 0){
	continue;   
      } else if (i != node_index) {
	int tentative_g = node_g+graph[row+i];
	int tentative_h = abs(end_node-i);
	NodeAStar * neighbor = new NodeAStar(i,tentative_g, tentative_h);
	int neighbor_index = find_neighbor(open,open_size,i);
	printf("n_i: %d, i: %d, g: %d, h:%d\n",neighbor_index, i, tentative_g, tentative_h);
	// neighbor not in open
	if (neighbor_index == -1){
	  open.push_back(neighbor);
	  open_set.insert(i);
	  open_size++;
	}
	else if(tentative_g >= open[neighbor_index] -> g) {
	  continue;
	}
	else {
	  open[neighbor_index] = neighbor;
	}
      }
    }
  }
  return -1;
}

class Node {
 public:
    int index = 0;
    int weight = 0;
  Node(int n_index, int n_weight) {
    index = n_index;
    weight = n_weight;
  };
};
int dfs(int * graph, int start_node, int end_node, int no_of_nodes){
  stack<Node*> neighbors;
  set<int> visited;
  Node * start = new Node(start_node, 0);
  neighbors.push(start);
  while (!neighbors.empty()){
    Node* node = neighbors.top();
    int node_index = node-> index;
    int node_weight = node -> weight;
    // printf("Node_Index: %d Node_Weight: %d\n",node_index, node_weight);
    // found end
    if(node_index == end_node)
      return node_weight;
    if (auto iter = visited.find(node_index); iter == visited.end()){
      int row = node_index*no_of_nodes;

      for(int i = 0; i < no_of_nodes; i++){
	// edge exists
	if (graph[row+i] != 0){
	  int neighbor_weight = node_weight + graph[row+i];
	  //	  printf("Node_Index: %d Node_Weight: %d\n",row+i, neighbor_weight);
	  Node* neighbor = new Node(i,neighbor_weight);
	  neighbors.push(neighbor);
	}
      }
    }
    printf("Size of stack: %d \n", neighbors.size());
    visited.insert(node_index);
    neighbors.pop();

  }
  // end can not be found
  return -1;
}
  
// For now, only return weight not path
int bfs(int * graph,int start_node,int end_node, int no_of_nodes){
  queue<Node*> neighbors;
  set<int> visited;
  Node* start = new Node(start_node, 0);
  neighbors.push(start);
  while (!neighbors.empty()){
    Node* node = neighbors.front();
    int node_index = node ->index;
    int node_weight = node -> weight;
    printf("Node_Index: %d Node_Weight: %d\n",node_index, node_weight);
    // end node found
    if (node_index == end_node)
      return node_weight;
    // node is not in visited
    if (auto iter = visited.find(node_index); iter == visited.end()){
      int row = node_index*no_of_nodes;

      for (int i = 0; i < no_of_nodes; i ++) {
	// edge exists
	if (graph[row+i] != 0){
	  int neighbor_weight = node_weight+graph[row+i];
	  Node* neighbor = new Node(i,neighbor_weight);
	  neighbors.push(neighbor);
	}
      }
    }
    printf("Size of queue: %d \n",neighbors.size());
    visited.insert(node_index);
    neighbors.pop();    

  }
  // end could not be reached
  return -1;
}

// adjacency matrix
void init_graph(int* graph, int no_of_nodes) {
  int rand_online[] = {8 ,7 ,1 ,6 ,9 ,4 ,7 ,4 ,6 ,9,7 ,4 ,2 ,6 ,2 ,5 ,7 ,4 ,0 ,6,1 ,2 ,5 ,2 ,4 ,5 ,3 ,2 ,4 ,3,6 ,6 ,2 ,1 ,3 ,4 ,9 ,1 ,6 ,6,9 ,2 ,4 ,3 ,1 ,8 ,7 ,9 ,1 ,9,4 ,5 ,5 ,4 ,8 ,6 ,6 ,0 ,5 ,5,7 ,7 ,3 ,9 ,7 ,6 ,9 ,5 ,2 ,1,4 ,4 ,2 ,1 ,9 ,0 ,5 ,8 ,7 ,9,6 ,0 ,4 ,6 ,1 ,5 ,2 ,7 ,2 ,8,9 ,6 ,3 ,6 ,9 ,5 ,1 ,9 ,8 ,7};
  int six[] = {0, 2, 5, 1, 0, 0, 2, 0, 3, 2, 0, 0, 5, 3, 0, 3, 1, 5, 1, 2, 3, 0, 1, 0, 0, 0, 1, 1, 0, 2, 0, 0, 5, 0, 2, 0};
  for (int i = 0; i < no_of_nodes; i ++){
    for (int j = 0; j < no_of_nodes; j++){
      // graph is bidirectional and has no reflexive edges
      int first = i*no_of_nodes+j;
      int second = i+no_of_nodes*j;
      if (j > i) {
	//	graph[first] = rand() & (no_of_nodes-1);
	//	graph[second] = graph[first];
	graph[first] = six[first];
	graph[second] = graph[first];
      }
    }
  }
  graph = rand_online;
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
  int no_of_nodes = NO_OF_NODES;
  int start_node = 0;
  int end_node = no_of_nodes-1;
  if (argc > 1) {
    int i = 1;
    no_of_nodes = *argv[i++];
    start_node = *argv[i++];
    end_node = *argv[i++];
  }
  if (start_node >= no_of_nodes || end_node >= no_of_nodes || start_node < 0 || end_node < 0) {
    printf("Error: start_node %d or end_node %d has to be valid node[0-%d]\n", start_node, end_node, no_of_nodes);
  }  
  graph = new int[no_of_nodes*no_of_nodes];

  init_graph(graph, no_of_nodes);
  output_graph(graph, no_of_nodes);
  printf("start : %d\n",start_node);
  printf("end : %d\n", end_node);
  int bfs_result = bfs(graph,start_node, end_node, no_of_nodes);
  printf("bfs_result : %d\n", bfs_result);
  int dfs_result = dfs(graph,start_node,end_node, no_of_nodes);
  printf("dfs_result : %d\n", dfs_result);
  int astar_result = astar(graph,start_node, end_node, no_of_nodes);
  printf("astar_result : %d\n", astar_result);
  return 0;
}
