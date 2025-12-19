#include <cuda_runtime.h>

static inline float get_value(
    const float* input,
    int b, int c, int y, int x,
    int B, int C, int H, int W)
{
    if (x < 0 || x >= W || y < 0 || y >= H)
        return 0.0f;

    int idx = ((b * C + c) * H + y) * W + x;
    return input[idx];
}

void grid_sample_serial(
    const float* input,
    const float* grid,
    float* output,
    int B, int C,
    int H, int W,
    int H_out, int W_out)
{
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {

                int gidx = ((b * H_out + h) * W_out + w) * 2;
                float x_norm = grid[gidx + 0];
                float y_norm = grid[gidx + 1];

                //aligning corners
                float x = (x_norm + 1.0f) * 0.5f * (W - 1);
                float y = (y_norm + 1.0f) * 0.5f * (H - 1);

                int x0 = (int)floorf(x);
                int y0 = (int)floorf(y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float wx = x - x0;
                float wy = y - y0;

                for (int c = 0; c < C; ++c) {
                    float v00 = get_value(input, b, c, y0, x0, B, C, H, W);
                    float v01 = get_value(input, b, c, y0, x1, B, C, H, W);
                    float v10 = get_value(input, b, c, y1, x0, B, C, H, W);
                    float v11 = get_value(input, b, c, y1, x1, B, C, H, W);

                    float val =
                        (1.0f - wx) * (1.0f - wy) * v00 +
                        wx          * (1.0f - wy) * v01 +
                        (1.0f - wx) * wy          * v10 +
                        wx          * wy          * v11;

                    int out_idx =
                        ((b * C + c) * H_out + h) * W_out + w;

                    output[out_idx] = val;
                }
            }
        }
    }
}
