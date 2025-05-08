#ifndef COMMON_FUNCTIONS
#define COMMON_FUNCTIONS
#include <climits>
int astar_cuda(uint64_t * openlist, uint64_t * closedList, uint64_t * src, uint64_t * dst, uint64_t * gList, uint64_t * hList, uint64_t * parent, uint64_t * nVV, int my_rank, uint64_t my_work, uint64_t srcPtrs_size, uint64_t dst_size, uint64_t grid_size, uint64_t no_of_nodes, int nprocs, uint64_t heuristic_type, uint64_t end, int iteration);

#endif /* COMMON_FUNCTIONS */
