clear
#nvcc -Xptxas -O3 -O3 -arch=sm_75 -I./cnpy dat_cuda/main.cu -o dat_cuda/exe
nvcc -Xptxas -O3 -arch=sm_75 dat_cuda/main.cu cnpy/cnpy.cpp -I./cnpy -o dat_cuda/exe -lz

./dat_cuda/exe