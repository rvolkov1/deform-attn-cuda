#clear
##nvcc -Xptxas -O3 -O3 -arch=sm_75 -I./cnpy dat_cuda/main.cu -o dat_cuda/exe
#nvcc -Xptxas -O3 -arch=sm_75 dat_cuda/main.cu cnpy/cnpy.cpp -I./cnpy -o dat_cuda/exe   -I$CUDNN_HOME/include \
#  -L$CUDNN_HOME/lib \
#  -lz -lcudnn
#
#./dat_cuda/exe

g++ -O3 -c cnpy/cnpy.cpp -I./cnpy -o cnpy.o

nvcc -O3 -arch=sm_75 dat_cuda/main.cu cnpy.o \
  -I./cnpy \
  -I$CUDNN_HOME/include \
  -L$CUDNN_HOME/lib \
  -L$CUDA_HOME/lib64 \
  -I./cudnn-frontend/include \
  -lcudnn -lz \
  -lcublas \
  -Xlinker -rpath -Xlinker $CUDNN_HOME/lib \
  -o dat_cuda/exe

./dat_cuda/exe

