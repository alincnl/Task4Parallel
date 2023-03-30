#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using namespace std;

__global__ void update(double* A, double* Anew, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < size - 1 && j > 0 && i > 0 && i < size - 1)
    {
       Anew[i * size + j] = 0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j] + A[i * size + j - 1] + A[i * size + j + 1]);
    }
}

__global__ void substract(double* A, double* Anew, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i >= 0 && i < size && j >= 0 && j < size)
		A[i*size + j] = Anew[i*size + j] - A[i*size + j];
}

int main(void)
{
    auto begin = std::chrono::steady_clock::now();

    double tol = atof(argv[1]);
    int size = atoi(argv[2]), iter_max = atoi(argv[3]);

    double* A = new double[size*size];
    double* Anew = new double[size*size];

    double *d_A = NULL, *d_Anew = NULL;

    cudaError_t cudaerr = cudaSuccess;
    cudaerr = cudaMalloc((void **)&d_A, sizeof(double)*size*size);
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    cudaerr = cudaMalloc((void **)&d_Anew, sizeof(double)*size*size);
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    int iter = 0;
    double error = 1;
    double add = 10.0 / (size - 1);

    A[0] = 10;
    A[size - 1] = 20;
    A[(size - 1)*(size)] = 20;
    A[(size - 1)*(size)+ size - 1] = 30;
    Anew[0] = 10;
    Anew[size - 1] = 20;
    Anew[(size - 1)*(size)] = 20;
    Anew[(size - 1)*(size)+ size - 1] = 30;

	for (size_t i = 1; i < size - 1; i++) {
		A[i] = A[i - 1] + add;
        A[(size - 1)*(size)+i] = A[(size - 1)*(size)+i - 1] + add;
        A[i*(size)] = A[(i - 1) *(size)] + add;
        A[i*(size)+size - 1] = A[(i - 1)*(size)+size - 1] + add;
        Anew[i] = A[i - 1] + add;
        Anew[(size - 1)*(size)+i] = A[(size - 1)*(size)+i - 1] + add;
        Anew[i*(size)] = A[(i - 1) *(size)] + add;
        Anew[i*(size)+size - 1] = A[(i - 1)*(size)+size - 1] + add;
	}

    cudaMemcpy(d_A, A, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew, size*size*sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadPerBlock = dim3(32, 32);
    dim3 blocksPerGrid = dim3(ceil(size/threadPerBlock.x),ceil(size/threadPerBlock.y));

    double* d_error;
    cudaMalloc(&d_error, sizeof(double));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_A, d_error, size*size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    while((error > tol) && (iter < iter_max)) 
    {
        iter = iter + 1;
        if ((iter % 100 == 0) or (iter == iter_max) or (iter==1)) {
            update<<<blocksPerGrid, threadPerBlock>>>(d_A, d_Anew, size);
            substract<<<blocksPerGrid, threadPerBlock>>>(d_A, d_Anew, size);

            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_A, d_error, size*size);
            
            cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << iter << ":" << error << "\n";
        }
        else{  
            update<<<blocksPerGrid, threadPerBlock>>>(d_A, d_Anew, size);
            substract<<<blocksPerGrid, threadPerBlock>>>(d_A, d_Anew, size);
        }
        cudaMemcpy(d_A, d_Anew, size*size*sizeof(double), cudaMemcpyDeviceToDevice);
    }

    std::cout << iter << ":" << error << "\n";

    delete[] A;
    delete[] Anew;

    cudaFree(d_A);
    cudaFree(d_Anew);
    cudaFree(d_error);

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
    std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    return 0;
}
