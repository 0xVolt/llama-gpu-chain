#include <stdio.h>

__global__ void testGPU() {
    printf("This runs on the GPU!\n");
}

int main() {
    printf("This runs on the CPU!\n");

    testGPU<<<1, 1>>>();

    cudaDeviceSynchronize();

    return (0);
}