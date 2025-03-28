#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define KERNEL_SOURCE "" \
"__kernel void edge_detection(__global const uchar *input, __global uchar *output, int width, int height) {" \
"    int x = get_global_id(0);" \
"    int y = get_global_id(1);" \
"    int gx = 0, gy = 0;" \
"    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};" \
"    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};" \
"    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {" \
"        for (int i = -1; i <= 1; i++) {" \
"            for (int j = -1; j <= 1; j++) {" \
"                int pixel = input[(y + j) * width + (x + i)];" \
"                gx += sobel_x[j + 1][i + 1] * pixel;" \
"                gy += sobel_y[j + 1][i + 1] * pixel;" \
"            }" \
"        }" \
"        output[y * width + x] = (uchar)clamp(sqrt((float)(gx * gx + gy * gy)), 0.0f, 255.0f);" \
"    } else {" \
"        output[y * width + x] = 0;" \
"    }" \
"}"

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 512, height = 512;
    size_t imgSize = width * height * sizeof(unsigned char);
    unsigned char *input = (unsigned char *)malloc(imgSize);
    unsigned char *output = (unsigned char *)malloc(imgSize);

    
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem inputBuffer, outputBuffer;
    
    cl_int err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "clGetPlatformIDs");
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    checkError(err, "clGetDeviceIDs");
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");
    
    program = clCreateProgramWithSource(context, 1, (const char**)&KERNEL_SOURCE, NULL, &err);
    checkError(err, "clCreateProgramWithSource");
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    checkError(err, "clBuildProgram");
    
    kernel = clCreateKernel(program, "edge_detection", &err);
    checkError(err, "clCreateKernel");
    
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgSize, input, &err);
    checkError(err, "clCreateBuffer input");
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, NULL, &err);
    checkError(err, "clCreateBuffer output");
    
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    checkError(err, "clSetKernelArg");
    
    size_t global_size[2] = {width, height};
    size_t local_size[2] = {8, 8}; // Small work-group size for CPU
    
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(err, "clEnqueueNDRangeKernel");
    
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imgSize, output, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer");
    
    
    printf("Output pixels:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", output[i]);
    }

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(input);
    free(output);
    
    return 0;
}
