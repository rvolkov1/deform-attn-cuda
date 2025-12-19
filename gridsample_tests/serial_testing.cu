#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "serial.cuh"

static double elapsed_ms(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) * 1000.0 +
           (b.tv_nsec - a.tv_nsec) / 1e6;
}

int main(void)
{
    /* Problem sizes (match CUDA tests) */
    int B = 4;
    int C = 32;
    int H = 32;
    int W = 32;
    int H_out = 256;
    int W_out = 256;

    int input_size  = B * C * H * W;
    int grid_size   = B * H_out * W_out * 2;
    int output_size = B * C * H_out * W_out;

    float* input  = (float*)malloc(input_size  * sizeof(float));
    float* grid   = (float*)malloc(grid_size   * sizeof(float));
    float* output = (float*)malloc(output_size * sizeof(float));

    if (!input || !grid || !output) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize input */
    for (int i = 0; i < input_size; ++i)
        input[i] = (float)(i % 127) / 127.0f;

    /* Initialize grid */
    for (int b = 0; b < B; ++b)
        for (int y = 0; y < H_out; ++y)
            for (int x = 0; x < W_out; ++x) {
                int idx = ((b * H_out + y) * W_out + x) * 2;
                grid[idx + 0] = (2.0f * x) / (W_out - 1) - 1.0f;
                grid[idx + 1] = (2.0f * y) / (H_out - 1) - 1.0f;
            }


    printf("========\nGrid Sample Baseline\n");


    /* Warmup */
    for (int i = 0; i < 2; ++i)
        grid_sample_serial(input, grid, output,
                           B, C, H, W, H_out, W_out);

    /* Timing */
    int iters = 5;
    struct timespec t0, t1;

    printf("Running \n");

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i)
        grid_sample_serial(input, grid, output,
                           B, C, H, W, H_out, W_out);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = elapsed_ms(t0, t1);
    double avg_ms = ms / iters;

    printf("Average kernel time: %.4f ms\n", avg_ms);

    free(input);
    free(grid);
    free(output);

    return 0;
}
