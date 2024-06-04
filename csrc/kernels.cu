#include "kernels.h"

__global__ void add_kernel(int a, int b, int *c) {
    *c = a + b + 12;
}

int add(int a, int b) {
    int c;
    int *dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int));
    add_kernel<<<1, 1>>>(a, b, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    return c;
}
