set -e

nvcc -O3 serial_testing.cu -o test_serial

./test_serial

nvcc -O3 naive_parallel_testing.cu -o test_naive_parallel

./test_naive_parallel

nvcc -O3 shared_parallel_testing.cu -o test_shared_parallel

./test_shared_parallel