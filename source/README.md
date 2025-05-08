## Directory Structure
## Make sure that igraph is installed in the machines
source/

|- bfs_serial

   |- bfs_serial.cpp
   
   |- bfs_serial

|- bfs_mpi

   |- bfs_mpi_no_cuda.cpp
   
   |- bfs_mpi_no_cuda

|- bfs_cuda

   |- bfs_cuda_no_mpi.cu
   
   |- bfs_cuda_no_mpi

|- bfs_mpi_cuda

   |- bfs_mpi.cpp
   
   |- bfs_cuda.cu
   
   |- bfs_mpi.o
   
   |- bfs_cuda.o
   
   |- common_functions_bfs.h
   
   |- bfs_mpi_cuda

|- astar_serial

   |- astar_serial.cpp
   
   |- astar_serial

|- bfs_mpi_path

   |- bfs_mpi_no_cuda_path.cpp
   
   |- bfs_mpi_no_cuda_path
   
   |- output.dot

|- bfs_cuda_path

   |- bfs_cuda_no_mpi_path.cu
   
   |- bfs_cuda_no_mpi_path
   
   |- output.dot

|- bfs_mpi_cuda_path

   |- bfs_mpi_path.cpp
   
   |- bfs_cuda_path.cu
   
   |- bfs_mpi_path.o
   
   |- bfs_cuda_path.o
   
   |- common_functions_bfs_path.h
   
   |- bfs_mpi_cuda_path
   
   |- output.dot

|- Common/  

|- Makefile

|- README.md

### BFS Serial
Compile:
make bfs_serial/bfs_serial
Usage:
cd bfs_serial
./bfs_serial <no_of_nodes> <start_node> <end_node> <percent>
|- no_of_nodes <= 46340
|- start_node bet. 0 and no_of_nodes
|- end_node bet. 0 and no_of_nodes
|- percent - expected number of edges (no_of_nodes^2*percent)
Samples:

./bfs_serial 10000 0 9999 0.0003
./bfs_serial 1000 0 999 0.003
./bfs_serial 100 0 99 0.03

### BFS MPI No CUDA
Compile:
make bfs_mpi/bfs_mpi_no_cuda
Usage:
cd bfs_mpi
mpirun -n <nprocs> bfs_mpi_no_cuda <no_of_nodes> <start_node> <end_node> <percent>
|- no_of_nodes <= 46340
|- start_node bet. 0 and no_of_nodes
|- end_node bet. 0 and no_of_nodes
|- percent - expected number of edges (no_of_nodes^2*percent)
Samples:

mpirun -n 4 bfs_mpi_no_cuda 10000 0 9999 0.0003
mpirun -n 3 bfs_mpi_no_cuda 1000 0 999 0.003
mpirun -n 2 bfs_mpi_no_cuda 100 0 99 0.03

### BFS CUDA No MPI
Compile:
make bfs_cuda/bfs_cuda_no_mpi
Usage:
cd bfs_cuda
./bfs_cuda_no_mpi <no_of_nodes> <start_node> <end_node> <percent>
|- no_of_nodes <= 46340
|- start_node bet. 0 and no_of_nodes
|- end_node bet. 0 and no_of_nodes
|- percent - expected number of edges (no_of_nodes^2*percent)
Samples:

./bfs_cuda_no_mpi 10000 0 9999 0.0003
./bfs_cuda_no_mpi 1000 0 999 0.003
./bfs_cuda_no_mpi 100 0 99 0.03


### BFS MPI CUDA
Compile:
make bfs_mpi_cuda/bfs_mpi_cuda
Usage:
cd bfs_mpi_cuda
mpirun -n <nprocs> bfs_mpi_cuda <no_of_nodes> <start_node> <end_node> <percent>
|- no_of_nodes <= 46340
|- start_node bet. 0 and no_of_nodes
|- end_node bet. 0 and no_of_nodes
|- percent - expected number of edges (no_of_nodes^2*percent)
Samples:

mpirun -n 4 bfs_mpi_cuda 10000 0 9999 0.0003
mpirun -n 3 bfs_mpi_cuda 1000 0 999 0.003
mpirun -n 2 bfs_mpi_cuda 100 0 99 0.03

### ASTAR SERIAL
Compile:
make astar_serial/astar_serial
Usage:
cd astar_serial
./astar_serial <grid_size> <start> <end> <percent> <obstacle_type> <heuristic>
|- grid_size - largest I've tried is 1000 or 1000000 nodes (no_of_nodes = grid_size * grid_size)
|- start bet. 0 and no_of_nodes
|- end bet. 0 and no_of_nodes
|- percent - likelihood of a block in the grid
|- obstacle_type - 1 is WALLS and 0 is DOTS
|- heuristic - manhattan or euclidean
Samples:

./astar_serial 100 0 9999 0.05 1 manhattan
./astar_serial 50 0 2499 0.05 1 manhattan
./astar_serial 10 0 99 0.05 1 manhattan

###BFS MPI No CUDA PATH
Compile:
make bfs_mpi_path/bfs_mpi_no_cuda_path
Usage:
cd bfs_mpi_path
#### Lattice
mpirun -n <nprocs> bfs_mpi_no_cuda_path <grid_size> <start> <end>
|- grid_size - largest I've tried is 7080 or 50126400 nodes (no_of_nodes = grid_size * grid_size)
|- start bet. 0 and no_of_nodes
|- end bet. 0 and no_of_nodes

#### LGL file - opte-out.lgl
mpirun -n <nprocs> bfs_mpi_no_cuda_path <start> <end>
|- start bet. 0 and 558328
|- end bet. 0 and 558328

Samples:
mpirun -n 4 bfs_mpi_no_cuda_path 1000 0 999999
mpirun -n 3 bfs_mpi_no_cuda_path 100 0 9999
mpirun -n 2 bfs_mpi_no_cuda_path 10 0 99
mpirun -n 2 bfs_mpi_no_cuda_path 0 500000

###BFS CUDA No MPI PATH
Compile:
make bfs_cuda_path/bfs_cuda_no_mpi_path
Usage:
cd bfs_cuda_path
#### Lattice
./bfs_cuda_no_mpi_path <grid_size> <start> <end>
|- grid_size - largest I've tried is 7000 or 49000000 nodes (no_of_nodes = grid_size * grid_size)
|- start bet. 0 and no_of_nodes
|- end bet. 0 and no_of_nodes

#### LGL file - opte-out.lgl
./bfs_cuda_no_mpi_path <start> <end>
|- start bet. 0 and 558328
|- end bet. 0 and 558328

Samples:
./bfs_cuda_no_mpi_path 1000 0 999999
./bfs_cuda_no_mpi_path 100 0 9999
./bfs_cuda_no_mpi_path 10 0 99
./bfs_cuda_no_mpi_path 0 558328

###BFS CUDA MPI PATH
Compile:
make bfs_mpi_cuda_path/bfs_mpi_cuda_path
Usage:
cd bfs_mpi_cuda_path
mpirun -n <nprocs> bfs_mpi_cuda_path <grid_size> <start> <end>
|- grid_size - largest I've tried is 7000 or 49000000 (no_of_nodes = grid_size * grid_size)
|- start bet. 0 and no_of_nodes
|- end bet. 0 and no_of_nodes

#### LGL file - opte-out.lgl
mpirun -n <nprocs> bfs_mpi_cuda_path <start> <end>
|- start bet. 0 and 558328
|- end bet. 0 and 558328

Samples:
mpirun -n 4 bfs_mpi_cuda_path 1000 0 999999
mpirun -n 3 bfs_mpi_cuda_path 100 0 9999
mpirun -n 2 bfs_mpi_cuda_path 10 0 99
mpirun -n 2 bfs_mpi_cuda_path 0 500000


