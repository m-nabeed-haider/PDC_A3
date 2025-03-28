#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#define KERNEL_SIZE 3

const char *kernelSource = R"(
__kernel void convolve(
    __global float *input, 
    __global float *output, 
    __constant float *kernel, 
    const int width, 
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int half_k = 1; // Kernel size is 3x3

    float sum = 0.0f;
    for (int ky = -half_k; ky <= half_k; ky++) {
        for (int kx = -half_k; kx <= half_k; kx++) {
            int nx = x + kx;
            int ny = y + ky;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx] * kernel[(ky + half_k) * KERNEL_SIZE + (kx + half_k)];
            }
        }
    }
    output[y * width + x] = sum;
}
)";

void checkErr(cl_int err, const char *name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 512, height = 512;
    std::vector<float> input(width * height, 1.0f);  
    std::vector<float> output(width * height, 0.0f);
    float kernel[KERNEL_SIZE * KERNEL_SIZE] = {1, 0, -1, 1, 0, -1, 1, 0, -1};

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_func;
    cl_mem input_buf, output_buf, kernel_buf;

    cl_int err;
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkErr(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    checkErr(err, "clGetDeviceIDs");

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkErr(err, "clCreateContext");

    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkErr(err, "clCreateCommandQueueWithProperties");

    input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * sizeof(float), input.data(), &err);
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), nullptr, &err);
    kernel_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), kernel, &err);
    checkErr(err, "clCreateBuffer");

    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    checkErr(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        checkErr(err, "clBuildProgram");
    }

    kernel_func = clCreateKernel(program, "convolve", &err);
    checkErr(err, "clCreateKernel");

    err = clSetKernelArg(kernel_func, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(kernel_func, 1, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(kernel_func, 2, sizeof(cl_mem), &kernel_buf);
    err |= clSetKernelArg(kernel_func, 3, sizeof(int), &width);
    err |= clSetKernelArg(kernel_func, 4, sizeof(int), &height);
    checkErr(err, "clSetKernelArg");

    size_t global_work_size[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    auto start = std::chrono::high_resolution_clock::now();

    err = clEnqueueNDRangeKernel(queue, kernel_func, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    checkErr(err, "clEnqueueNDRangeKernel");
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, width * height * sizeof(float), output.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueReadBuffer");

    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseMemObject(kernel_buf);
    clReleaseKernel(kernel_func);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}