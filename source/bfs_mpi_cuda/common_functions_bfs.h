#ifndef COMMON_FUNCTIONS
#define COMMON_FUNCTIONS

int bfs_cuda(int * srcPtrs, int * dst, int * level, int * nVV, int my_rank, int my_work, int currLevel, int srcPtrs_size, int dst_size, int no_of_nodes, int nprocs);

#endif /* COMMON_FUNCTIONS */
