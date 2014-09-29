#ifndef STREAM_H
#define STREAM_H

#include <cuda.h>
#include <stdio.h>

#define BlockSize 1024

void GPU_naive_prefix_sum(float * in, float * out, int n);
void GPU_single_shared_prefix_sum(float * in, float * out, int n);
void GPU_shared_prefix_sum(float * in, float * out, int n);
void GPU_scatter(float * in, float * out, int n);

#endif