# set -e
# #note the file paths are a little different in this file than in run_cu.sh
# #(they go up one folder level for cnpy because cnpy is in a different folder)

# g++ -O3 -c ../cnpy/cnpy.cpp -I../cnpy -o cnpy.o

# nvcc -O3 -arch=sm_75 test_layernorm.cu cnpy.o \
#   -I../cnpy \
#   -L$CUDA_HOME/lib64 \
#   -lz \
#   -o test_layernorm_exe

# ./test_layernorm_exe

#!/bin/bash
set -e

# Note the file paths are adjusted for cnpy location
# Compile cnpy if not already compiled (or recompile for safety)
g++ -O3 -c ../cnpy/cnpy.cpp -I../cnpy -o cnpy.o

# Compile CUDA test suite
nvcc -O3 -arch=sm_75 test_layernorm.cu cnpy.o \
  -I../cnpy \
  -L$CUDA_HOME/lib64 \
  -lz \
  -o test_layernorm_exe

for i in {1..10}; do
    ./test_layernorm_exe test$i
done