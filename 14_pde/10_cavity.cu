#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

#define BLOCK_SIZE 16


#define CUDA_CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}


// Initialize speed and pressure arrays
void initArrays(float* u, float* v, float* p, float* b, int nx, int ny)
{
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            u[j*nx + i] = 0;
            v[j*nx + i] = 0;
            p[j*nx + i] = 0;
            b[j*nx + i] = 0;
        }
    }
}

// Calculate speed and pressure
__global__ void computeVelocityAndPressure(float* u, float* v, float* p, float* b, int nx, int ny,
                                           float dx, float dy, float dt, float rho, float nu, int nit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1)
    {
        for (int t = 0; t < nit; t++)
        {
            float pn = p[j*nx + i];

            
            p[j*nx + i] = (dy*dy*(pn*nx + i+1 + pn*nx + i-1) + dx*dx*(pn*nx + i+nx + pn*nx + i-nx) -
                           b[j*nx + i]*dx*dx*dy*dy) / (2*(dx*dx + dy*dy));

            
            if (i == nx-1)
                p[j*nx + i] = p[j*nx + i-1];
            if (i == 0)
                p[j*nx + i] = p[j*nx + i+1];
            if (j == ny-1)
                p[j*nx + i] = 0;
            if (j == 0)
                p[j*nx + i] = p[(j+1)*nx + i];
        }

        u[j*nx + i] = u[j*nx + i] - u[j*nx + i] * dt / dx * (u[j*nx + i] - u[j*nx + i-1]) -
                      v[j*nx + i] * dt / dy * (u[j*nx + i] - u[(j-1)*nx + i]) -
                      dt / (2 * rho * dx) * (p[j*nx + i+1] - p[j*nx + i-1]) +
                      nu * dt / (dx*dx) * (u[j*nx + i+1] - 2*u[j*nx + i] + u[j*nx + i-1]) +
                      nu * dt / (dy*dy) * (u[(j+1)*nx + i] - 2*u[j*nx + i] + u[(j-1)*nx + i]);

        v[j*nx + i] = v[j*nx + i] - u[j*nx + i] * dt / dx * (v[j*nx + i] - v[j*nx + i-1]) -
                      v[j*nx + i] * dt / dy * (v[j*nx + i] - v[(j-1)*nx + i]) -
                      dt / (2 * rho * dy) * (p[(j+1)*nx + i] - p[(j-1)*nx + i]) +
                      nu * dt / (dx*dx) * (v[j*nx + i+1] - 2*v[j*nx + i] + v[j*nx + i-1]) +
                      nu * dt / (dy*dy) * (v[(j+1)*nx + i] - 2*v[j*nx + i] + v[(j-1)*nx + i]);
    }
}


int main()
{
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    float dx = 2.0 / (nx - 1);
    float dy = 2.0 / (ny - 1);
    float dt = 0.01;
    float rho = 1.0;
    float nu = 0.02;

    int size = nx * ny * sizeof(float);


    float* u_host = (float*)malloc(size);
    float* v_host = (float*)malloc(size);
    float* p_host = (float*)malloc(size);
    float* b_host = (float*)malloc(size);
    float* u_dev, *v_dev, *p_dev, *b_dev;


    initArrays(u_host, v_host, p_host, b_host, nx, ny);


    CUDA_CHECK(cudaMalloc((void**)&u_dev, size));
    CUDA_CHECK(cudaMalloc((void**)&v_dev, size));
    CUDA_CHECK(cudaMalloc((void**)&p_dev, size));
    CUDA_CHECK(cudaMalloc((void**)&b_dev, size));

    CUDA_CHECK(cudaMemcpy(u_dev, u_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_dev, v_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p_dev, p_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_dev, b_host, size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  
    for (int n = 0; n < nt; n++)
    {
        computeVelocityAndPressure<<<grid, block>>>(u_dev, v_dev, p_dev, b_dev, nx, ny,
                                                   dx, dy, dt, rho, nu, nit);


        CUDA_CHECK(cudaMemset(p_dev + (ny-1)*nx, 0, nx*sizeof(float)));
        CUDA_CHECK(cudaMemset(u_dev + (ny-1)*nx, 0, nx*sizeof(float)));
        CUDA_CHECK(cudaMemset(u_dev, 0, nx*sizeof(float)));
        CUDA_CHECK(cudaMemset(v_dev, 0, nx*sizeof(float)));


        CUDA_CHECK(cudaMemcpy(u_host, u_dev, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_host, v_dev, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(p_host, p_dev, size, cudaMemcpyDeviceToHost));


        // Display Results
        std::cout << "Step: " << n << std::endl;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::cout << "u(" << i << ", " << j << ") = " << u_host[j*nx + i] << ", ";
                std::cout << "v(" << i << ", " << j << ") = " << v_host[j*nx + i] << ", ";
                std::cout << "p(" << i << ", " << j << ") = " << p_host[j*nx + i] << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // free memory
    free(u_host);
    free(v_host);
    free(p_host);
    free(b_host);
    CUDA_CHECK(cudaFree(u_dev));
    CUDA_CHECK(cudaFree(v_dev));
    CUDA_CHECK(cudaFree(p_dev));
    CUDA_CHECK(cudaFree(b_dev));

    return 0;
}
