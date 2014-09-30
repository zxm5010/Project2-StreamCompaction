#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include "StreamCompaction.h"

using namespace std;
// part 2: naive prefix sum
__global__ void writeBuffer(float * in, float * out, int n)
{
	int index = blockDim.x * blockIdx.x +  threadIdx.x; 
	if(index < n )
		out[index] = in[index];
}

__global__ void naive_sum(float * in, float * out, int d, int n)
{
	int index = blockDim.x * blockIdx.x +  threadIdx.x;  

	if (index >= d && index < n)
		in[index] = out[index - d] + out[index];
}

__global__ void writeOutput(float * in, float * out, int n)
{
	int index = blockDim.x * blockIdx.x +  threadIdx.x; 
	if(index < n)
		out[index+1] = in[index];
}


// part 3a: shared sum with single block
__global__ void single_shared_sum(float * in, float * out, int n)
{
	extern __shared__ float tmp[];

	int index = threadIdx.x;  
	tmp[index] = in[index]; 
	__syncthreads();

	if(index < n)
	{
		for(int i = 1; i < n; i *=2)
		{
			if (index >= i)
			{
				tmp[index] = tmp[index] + tmp[index-i];	
			}
			__syncthreads();
		}
	}
	out[index+1] = tmp[index];
}


// part 3b: shared sum with arbitary array length
__global__ void shared_sum( float * in,  float * out, int n, float * sum)
{
	extern __shared__ float tmp[];

	int index = blockIdx.x * blockDim.x + threadIdx.x; 
	if(index < n)
		tmp[threadIdx.x] = in[index]; 

	if(index < n)
	{
		for(int i = 1; i < blockDim.x; i *=2)
		{
			__syncthreads();
			int j = (threadIdx.x + 1) * 2 * i -1;
			if (j < blockDim.x)
			{
				tmp[j] +=  tmp[j-i];	
			}
		}

		for(int i = BlockSize/4; i > 0; i /= 2)
		{
			__syncthreads();
			int j = (threadIdx.x + 1) * 2 * i -1;
			if (j + i < blockDim.x)
			{
				tmp[j + i] +=  tmp[j];	
			}
			
		}
	}

	__syncthreads();
	out[index+1] = tmp[threadIdx.x];

	if(threadIdx.x == 0)
		sum[blockIdx.x] = tmp[BlockSize - 1];
}

__global__ void add_sum( float * out, int n, float * sum)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < n && blockIdx.x > 0)
		out[index] += sum[blockIdx.x];
}

// part 4: scatter and stream compact
__global__ void generateBoolArray(float * in, float * out, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < n)
	{
		if(in[index] > 0)
			out[index] = 1;
		else
			out[index] = 0;
	}
}

__global__ void scatter(float * in, float * indexArray, float * out, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < n)
	{	
		if(in[index] > 0)
		{
			int tmp = indexArray[index];
			out[tmp] = in[index];
		}
	}
}


// wrapper fuctions

void GPU_naive_prefix_sum(float * in, float * out, int n)
{
	float * dev_in;
	float * dev_out;
	cudaMalloc((void**)&dev_in, n*(sizeof(float)));
	cudaMalloc((void**)&dev_out, n*(sizeof(float)));

	cudaMemcpy(dev_in, in, n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	for(int i = 1; i < n ; i *= 2)
	{
		writeBuffer<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_in, dev_out, n);
		naive_sum<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_in, dev_out, i, n);
	}
	writeOutput<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_in, dev_out, n);


	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	cout << "#GPU naive prefix sum performance:  "<< time << " ms" << endl;

	cudaMemcpy(out, dev_out, n * sizeof(float), cudaMemcpyDeviceToHost);
	out[0] = 0;
	cudaFree(dev_in);
	cudaFree(dev_out);
}

void GPU_single_shared_prefix_sum(float * in, float * out, int n)
{
	float * dev_in;
	float * dev_out;
	cudaMalloc((void**)&dev_in, n*(sizeof(float)));
	cudaMalloc((void**)&dev_out, n*(sizeof(float)));

	cudaMemcpy(dev_in, in, n * sizeof(float), cudaMemcpyHostToDevice);

	single_shared_sum<<<1, n, n * sizeof(float)>>>(dev_in, dev_out, n);

	cudaMemcpy(out, dev_out, n * sizeof(float), cudaMemcpyDeviceToHost);
	out[0] = 0;
	cudaFree(dev_in);
	cudaFree(dev_out);
}

void scan_sum(float * in, float * out, int n)
{
	dim3 fullBlocksPerGrid((int)ceil(float(n)/float(BlockSize)));

	for(int i = 1; i < n ; i *= 2)
	{
		writeBuffer<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(in, out, n);
		naive_sum<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(in, out, i, n);
	}
	writeOutput<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(in,  out, n);
}

void GPU_shared_prefix_sum(float * in, float * out, int n)
{
	float * dev_in;
	float * dev_out;
	float * dev_sum;
	int grid_size = (int)ceil(float(n)/float(BlockSize));

	cudaMalloc((void**)&dev_in, n*(sizeof(float)));
	cudaMalloc((void**)&dev_out, n*(sizeof(float)));
	cudaMalloc((void**)&dev_sum, grid_size * (sizeof(float)));

	cudaMemcpy(dev_in, in, n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 fullBlocksPerGrid(grid_size);

	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	shared_sum<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_in, dev_out, n, dev_sum);
	scan_sum(dev_sum, dev_sum, grid_size);
	add_sum<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_out,n,dev_sum);

	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	cout << "#GPU shared prefix sum performance:  "<< time << " ms" << endl;

	cudaMemcpy(out, dev_out, n * sizeof(float), cudaMemcpyDeviceToHost);
	out[0] = 0;

	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_sum);
}

void GPU_stream_compact(float * in, float * out, int n)
{
	float * dev_in;
	float * dev_out;
	float * dev_boolArray;
	float * dev_sum;
	int grid_size = (int)ceil(float(n)/float(BlockSize));

	cudaMalloc((void**)&dev_in, n*(sizeof(float)));
	cudaMalloc((void**)&dev_out, n*(sizeof(float)));
	cudaMalloc((void**)&dev_boolArray, n*(sizeof(float)));
	cudaMalloc((void**)&dev_sum, grid_size * (sizeof(float)));
	cudaMemcpy(dev_in, in, n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 fullBlocksPerGrid(grid_size);

	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	generateBoolArray<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_in, dev_boolArray, n);

	shared_sum<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_boolArray, dev_boolArray, n, dev_sum);
	scan_sum(dev_sum, dev_sum, grid_size);
	add_sum<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_boolArray,n,dev_sum);

	scatter<<<fullBlocksPerGrid, BlockSize, BlockSize * sizeof(float)>>>(dev_in, dev_boolArray, dev_out, n);


	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	cout << "#GPU stream compact performance:  "<< time << " ms" << endl;

	cudaMemcpy(out, dev_out, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_sum);
	cudaFree(dev_boolArray);
}

