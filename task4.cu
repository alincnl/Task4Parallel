#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using namespace std;
using namespace cub;

#define tol  1e-6

__global__ void update(double* A, double* Anew, int size)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (j < size - 1 && j > 0 && i > 0 && i < size - 1){
		double left = A[i * size + j - 1];
		double right = A[i * size + j + 1];
		double top = A[(i - 1) * size + j];
		double bottom = A[(i + 1) * size + j];
		Anew[i*size + j] = 0.25 * (left + right + top + bottom);
	}
}

__global__ void substract(double* A, double* Anew, double* res, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= 0 && i < size && j >= 0 && j < size)
		res[i*size + j] = Anew[i*size + j] - A[i*size + j];
}

__constant__ double add;

__global__ void fill(double* A, double* Anew, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size){
        A[i*(size) + 0] = 10 + add*i;
        A[i] = 10 + add*i;
        A[(size-1)*(size) + i] = 20 + add*i;
        A[i*(size)+size-1] = 20 + add*i;

        Anew[i*(size) + 0] = A[i*(size) + 0];
        Anew[i] = A[i];
        Anew[(size-1)*(size) + i] = A[(size-1)*(size) + i];
        Anew[i*(size)+size-1] = A[i*(size)+size-1];
    }
}

int main(int argc, char* argv[]){

    auto begin = std::chrono::steady_clock::now();
    cudaSetDevice(1);
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    const int size =512, iter_max = 1000;

    double *d_A = NULL, *d_Anew = NULL, *d_Aprev;

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

    cudaerr = cudaMalloc((void **)&d_Aprev, sizeof(double)*size*size);
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    int iter = 0;
    double error = 1;
    double addH = 10.0 / (size - 1);
    cudaMemcpyToSymbol(add, &addH, sizeof(double));

    dim3 threadPerBlock = dim3(32, 32);
    dim3 blocksPerGrid = dim3((size+threadPerBlock.x-1)/threadPerBlock.x,(size+threadPerBlock.y-1)/threadPerBlock.y);
    
    fill<<<blocksPerGrid, threadPerBlock>>>(d_A, d_Anew, size);

    double* d_error;
    cudaMalloc(&d_error, sizeof(double));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_A, d_error, size*size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    while((error > tol) && (iter < iter_max/100)) {
        iter = iter + 2;
        if(!graphCreated)
	    {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(int i = 0; i<100;i++){
                update<<<blocksPerGrid, threadPerBlock,0,stream>>>(d_Anew,d_A, size);
                update<<<blocksPerGrid, threadPerBlock,0,stream>>>( d_A,  d_Anew,size);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated=true;
        }
       // swap = d_A;
       // d_Aprev=d_A;
        cudaGraphLaunch(instance, stream);
	    cudaStreamSynchronize(stream);

        substract<<<blocksPerGrid, threadPerBlock,0,stream>>>(d_A, d_Anew, d_Aprev, size);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Aprev, d_error, size*size,stream);
        cudaMemcpyAsync(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
       // cudaMemcpyAsync(d_A, d_Anew, size*size*sizeof(double), cudaMemcpyDeviceToDevice);
        //swap = d_A;
        std::cout << iter << ":" << error << "\n";

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken: %3.1f ms\n", elapsedTime);

    cudaFree(d_A);
    cudaFree(d_Anew);
    cudaFree(d_error);

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
    std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    return 0;
}
