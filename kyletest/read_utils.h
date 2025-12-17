#ifndef READ_UTILS_H
#define READ_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

static inline int parse_shape_line(const char* line, int* B, int* C, int* H, int* W) {
    // Look for 4 integers inside brackets
    const char* p = strchr(line, '[');
    if (!p) return 0;

    if (sscanf(p, "[%d,%d,%d,%d]", B, C, H, W) == 4 ||
        sscanf(p, "[%d, %d, %d, %d]", B, C, H, W) == 4)
        return 1;

    return 0;
}

static inline float* read_tensor_txt(const char* path,
                       size_t* out_count,
                       int* B, int* C, int* H, int* W) 
{
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        return NULL;
    }

    char line[2048];

    // -----------------------------------------------------
    // PASS 1: Find the shape line
    // -----------------------------------------------------
    *B = *C = *H = *W = -1;

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "shape")) {
            // Next lines might contain `[B, C, H, W]`
            if (fgets(line, sizeof(line), f)) {
                if (!parse_shape_line(line, B, C, H, W)) {
                    fprintf(stderr, "Error: cannot parse shape\n");
                    fclose(f);
                    return NULL;
                }
            }
            break;
        }
    }

    if (*B <= 0 || *C <= 0 || *H <= 0 || *W <= 0) {
        fprintf(stderr, "Error: invalid shape values\n");
        fclose(f);
        return NULL;
    }

    // Go back to the top for numeric parsing
    rewind(f);

    // -----------------------------------------------------
    // PASS 2: Count floating-point numbers
    // -----------------------------------------------------
    size_t count = 0;
    int file_idx = 0;

    while (fgets(line, sizeof(line), f)) {
        file_idx += 1;
        if (file_idx < 6) continue;
        char* p = line;
        while (*p) {
            if (*p == '-' || isdigit(*p)) {
                char* end;
                strtof(p, &end);
                if (end != p) {
                    count++;
                    p = end;
                    continue;
                }
            }
            p++;
        }
    }

    // Allocate storage
    float* data = (float*)malloc(count * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: malloc failed\n");
        fclose(f);
        return NULL;
    }

    // -----------------------------------------------------
    // PASS 3: Parse floats into buffer
    // -----------------------------------------------------
    rewind(f);
    size_t idx = 0;
    file_idx = 0;

    while (fgets(line, sizeof(line), f)) {
        file_idx += 1;
        if (file_idx < 6) continue;
        char* p = line;
        while (*p) {
            if (*p == '-' || isdigit(*p)) {
                char* end;
                float v = strtof(p, &end);
                if (end != p) {
                    data[idx++] = v;
                    p = end;
                    continue;
                }
            }
            p++;
        }
    }

    fclose(f);

    *out_count = count;
    return data;
}

#endif