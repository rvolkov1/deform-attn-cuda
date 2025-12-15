clear
nvcc -Xptxas -O3 -O3 -arch=sm_75 dat_cuda/main.cu -o dat_cuda/exe -lcudnn

./dat_cuda/exe