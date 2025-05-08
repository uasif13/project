#include <cmath>
#include <math.h>

#include <iostream>
#include <sys/time.h>
#include <random>
#include <climits>

using namespace std;
using std::cout;
using std::cerr;
using std::endl;

#define MAX_BUFFER_SIZE 1 << 25

void init_graph(int* graph,int no_of_nodes, float percent);

int astar(int* graph, int start_node, int end_node, int no_of_nodes);


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
    printf("%*d", 3, graph[i]);
    
  }
  printf("\n");
}

class Graph {
public:
  int * row;
  int * col;
  int * o_index;
  int * o_array;
  int obstacle_count;
  int grid_size;
  int start;
  int end;
  Graph(int p_grid_size, float percent, int p_start, int p_end, int obstacle_type) {
    random_device rd;
    mt19937 gen(rd());
    vector<int> v_index;
    uniform_real_distribution<> dis (0.0,1/percent);
    uniform_real_distribution<> orient (0.0,1/0.5);
    grid_size = p_grid_size;
    start = p_start;
    end = p_end;
    int row, col;
    for (int i = 0; i < grid_size * grid_size; i++) {

	if (dis(gen) < 1 ? true: false) {
	  v_index.push_back(i);
	  if (orient(gen) < 1 ? true: false) {
	    for (int j = 0; j < 5; j ++) {
	      col = i %grid_size;
	      if (col+j < grid_size)
		v_index.push_back(i+j);
	    }
	  }else {
	    for (int j = 0; j < 5; j++) {
	      row = i / grid_size;
	      if (row+j < grid_size)
		v_index.push_back(i+j*grid_size);		
	    }
	  }
	}				

    }
    obstacle_count = v_index.size();
    o_array = new int[grid_size*grid_size];
    for (int i = 0; i < obstacle_count; i ++) {
      if (v_index[i] != start && v_index[i] != end)
	o_array[v_index[i]] = 1;
    }
					       
  }
    Graph(int p_grid_size, float percent, int p_start, int p_end) {
  random_device rd;
  mt19937 gen(rd());
  vector<int> v_row;
  vector<int> v_col;
  vector<int> v_index;
  int row_index, col_index;
  grid_size = p_grid_size;
  uniform_real_distribution<> dis(0.0,1/percent);
  for (int i = 0; i < grid_size*grid_size; i++) {
    
    if (i != p_start && i != p_end) {
      if (dis(gen) < 1 ? true: false) {
      row_index = i / grid_size;
      col_index = i % grid_size;
      v_row.push_back(row_index);
      v_col.push_back(col_index);
      v_index.push_back(i);
      //  printf("r:%d c:%d i:%d p_start: %d p_end: %d condition: %d\n", row_index, col_index, i, p_start, p_end, condition);
      }
    }
  }
  obstacle_count = v_row.size();
  row = new int[obstacle_count];
  col = new int[obstacle_count];
  o_index = new int[obstacle_count];
  o_array = new int[grid_size*grid_size];
  start = p_start;
  end = p_end;
  printf("\n");
  for (int i = 0; i < obstacle_count; i++) {
      row[i] = v_row[i];
      col[i] = v_col[i];
      o_index[i] = v_index[i];
      o_array[o_index[i]] = 1;
      //printf("r:%d c:%d i:%d\n", v_row[i], v_col[i], v_index[i]);
  }    
  }
};
/*
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
int astar(int* graph, string heuristic){
  int start_x = graph -> start_x;
  int start_y = graph -> start_y;
  int end_x = graph -> end_x;
  int end_y = graph -> end_y;
  int grid_size = graph -> grid_size;
  int row = graph -> row;
  int col = graph -> col;
  int index = 
  NodeAStar * start = new NodeAStar(start_node, start_g, start_h);
  vector<NodeAStar*> open;
  set<int> open_set;
  open.push_back(start);
  open_set.insert(start_node);
  int open_size = 1;
  
  vector<NodeAStar*> closed;
  set<int> closed_set;
  int closed_size = 0;
  
    struct
  {
    bool operator()(const NodeAStar* l, const NodeAStar* r) const{return l->index>r-> index;}
  }customLess;
  set<NodeAStar *,customLess, allocator<NodeAStar*>> closed;
  set <NodeAStar *, customLess, allocator<NodeAStar*>>open;
  auto cmp = [](NodeAStar* left, NodeAStar*right){return left->g < right->g;}; 
  priority_queue<NodeAStar*, vector<NodeAStar*>, decltype(cmp)> open_pq(cmp);
  open_pq.push(start);
  open.push(start);

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
*/
bool checkNB(int index_f, int neighbor, int grid_size) {
  if (neighbor == -1) {
    if (index_f % grid_size == 0)
      return false;
    else
      return true;
  } else if (neighbor == 1) {
    if (index_f % grid_size == grid_size -1)
      return false;
    else
      return true;
  } else if (neighbor == -grid_size) {
    if (index_f/grid_size == 0)
      return false;
    else
      return true;
  } else {
    if (index_f/grid_size == grid_size -1)
      return false;
    else
      return true;
  }
  return false;
}

int cal_heuristic(int nb,int grid_size, int end, string heuristic ) {
  int row_diff = nb/grid_size - end/grid_size;
  if (row_diff < 0)
    row_diff *= -1;
  int col_diff = nb%grid_size - end%grid_size;
  if (col_diff < 0)
    col_diff *= -1;
  if (heuristic == "manhattan")
    return row_diff + col_diff;
  else
    return pow(row_diff*row_diff+col_diff*col_diff,0.5);
}

// heuristic= manhattan and euclidean
int astar(Graph * graph, string heuristic) {
  int * o_array;
  int grid_size;
  int start, end;

  o_array = graph -> o_array;
  grid_size = graph -> grid_size;

  start = graph -> start;
  end = graph -> end;
  
  int * openList = new int[grid_size*grid_size];
  int * closedList = new int[grid_size*grid_size];
  int * gList = new int[grid_size*grid_size];
  int * hList = new int[grid_size*grid_size];

  int index_f;
  int index_f_score;
  int g_score;
  int h_score;
  int nb;
  int * neighbor = new int [4];
  neighbor[0] = -1;
  neighbor[1] = 1;
  neighbor[2] = -1*grid_size;
  neighbor[3] = 1*grid_size;
  
  openList[start] = 1;
  int open_size = 1;
  int closed_size = 0;
  while (open_size != 0) {
    // lowest f
    int index_f = grid_size*grid_size;
    int index_f_score = INT_MAX;

    for (int i = 0; i < grid_size*grid_size; i++) {
      if (openList[i] == 1 && gList[i] + hList[i] < index_f_score) {
	index_f = i;
	index_f_score = gList[i] + hList[i];
      }
    }
    //printf("\nindex_f: %d score: %d\n", index_f, index_f_score);
    if (index_f == end) {
      return index_f_score;
    }
    openList[index_f] = 0;
    open_size --;
    closedList[index_f] = 1;
    closed_size ++;

    for (int i = 0; i < 4; i++) {
      nb = index_f+neighbor[i];
      if (!checkNB(index_f, neighbor[i], grid_size))
	continue;
 
      if (closedList[nb] == 1) {
	continue;
      }
      if (o_array[nb] == 1) {
	continue;
      }
      //printf("nb %d\n", nb);
      g_score = gList[index_f] + 1;
 
      if (openList[nb] == 0) {
	openList[nb] = 1;
	open_size ++;
      } else if (gList[nb] != 0 && g_score >= gList[nb]) {
	continue;
      }
      gList[nb] = g_score;
      hList[nb] = cal_heuristic(nb, grid_size, end, heuristic);
      //printf("g_score %d gList: %d hList: %d\n", g_score, gList[nb], hList[nb]);
    }
  }
  return -1;
}

/*
void init_wall(map<array<int>, 2>* v_obstacles, int index, int orient, int width, int grid_size) {
  int row = index/grid_size;
  int col = index%grid_size;
  for (int i = 1; i <= width; i++) {
    if (orient == 1) {
      if (row - i >= 0)
	v_obstacles -> push_back(index-i*grid_size);
      if (row + i < grid_size)
	v_obstacles -> push_back(index+i*grid_size);
    } else {
      if (col - i >= 0)
	v_obstacles ->push_back(index-i);
      if (col + i < grid_size)
	v_obstacles ->push_back(index+i);
    }
  }
  }*/


// COO
// dots + wall
/*
void init_graph(int * row, int * col, int grid_size, float percent, int obstacle_type) {
      // grid, 1 is unreachable and 0 is reachable
  // length = 1 + 2 * width
  int width = 2;
  //  map<<array<int, 2> , int>, int> m_obstacles;
  int row, col;
 random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0,1/0.5);
  int orient = dis(gen) < 1 ? 1 : 0;
  printf("obstacle type: %d\n", obstacle_type);
  dis(0.0,1/percent);
  for (int i = 0 ; i < grid_size*grid_size; i++ ) {
    if (diss(gen) < 1 ? 1: 0) {
      //  row = i \ grid_size;
      // col = i % grid_size;
      array<int,2> obstacle {row, col};
      if (auto search = m_obstacles.find(obstacle); search == m_obstacles.end(obstacle)){
	m_obstacles.insert(pair{obstacle, 0});
            if (obstacle_type == 1) {
	      init_wall(&m_obstacles,i,orient,width,grid_size);
	      printf("v_obstacles_size: %d\n", v_obstacles.size());
	    }
      }

      printf("v_obstacles_size: %d\n", v_obstacles.size());
    }
  }
  for (int i = 0; i < v_obstacles.size(); i++) {
    graph[i] = v_obstacles[i];
  }
 }
*/
// dots

// single source to single destination path
int main(int argc, char* argv[]) {

  int * row;
  int * col;
  int * o_index;
  int * o_array;
  int obstacle_count;
  Graph * graph;
  int grid_size;
  int start, end;
  int obstacle_type;
  float percent;
  // manhattan, euclidean
  string heuristic;

  long astar_start, astar_end, astar_elapsed;
  
  struct timeval timecheck;

  if (argc == 7) {
    int i = 1;
    grid_size = atoi(argv[i++]);
    start = atoi(argv[i++]);
    end = atoi(argv[i++]);
    percent = atof(argv[i++]);
    obstacle_type = atoi(argv[i++]);
    heuristic = argv[i++];
  }
  else {
    grid_size = 10;
    start = 0;
    end = grid_size*grid_size - 1;
    percent = 0.05;
    obstacle_type = 1;
    heuristic = "manhattan";
  }
  if (start>= grid_size*grid_size || end>= grid_size*grid_size || start< 0 || end< 0) {
    printf("Error: start %d end %d %d has to be valid node[0-%d]\n",start,end, grid_size );
  }
  if (percent > 1.0 || percent <= 0.0 || (obstacle_type != 0&& obstacle_type != 1))
    printf("Error: percent %d out of bounds, obstacle type is not 0 or 1", percent);
  if (obstacle_type == 0)
    graph = new Graph(grid_size, percent, start, end);
  else
    graph = new Graph(grid_size, percent, start, end, obstacle_type);
  // row = graph -> row;
  //col = graph -> col;
  //o_index = graph -> o_index;
  //obstacle_count = graph -> obstacle_count;
  o_array = graph -> o_array;
  //output_vec(row, obstacle_count);
  //output_vec(col, obstacle_count);
  //output_vec(o_index, obstacle_count);
  //output_graph(o_array, grid_size);
  printf("start : %d end: %d\n",start, end);
  
  gettimeofday(&timecheck, NULL);
  astar_start = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;

  int astar_result = astar(graph,heuristic);
  printf("astar_result : %d\n", astar_result);

  gettimeofday(&timecheck, NULL);
  astar_end = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;

  astar_elapsed = astar_end - astar_start;

  printf("***********************\n");
  printf("no_of_nodes: %d time: %d\n", grid_size*grid_size, astar_elapsed);
    
  return 0;
}
