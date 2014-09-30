#include <iostream>
#include <stdlib.h>
#include "StreamCompaction.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust\copy.h>
#include <time.h>

using namespace std;

const int NUM = 10000;
const int SINGLE_SHARED_NUM = 500;

void CPU_prefix_sum(float * in, float * out, int n)
{
	out[0] = 0;
	for(int i = 1; i < n; ++i)
	{
		out[i] = out[i-1] + in[i-1];
	}
}

void CPU_scatter(float * in, float * out, int n)
{
	float * tmp = new float[n];
	for(int i = 0; i < n; ++i)
	{
		if(in[i]> 0)
			tmp[i] = 1;
		else
			tmp[i] = 0;
	}

	CPU_prefix_sum(tmp,out,n);
}

float * CPU_stream_compact(float * in, int n)
{
	float * tmp = new float[n];
	int compact_size = 0;
	CPU_scatter(in,tmp,n);

	compact_size = tmp[n-1] + 1;
	float * ret = new float[compact_size];


	for(int i = 0; i < n; ++i)
	{
		if(in[i] > 0)
		{
			int index = tmp[i];
			ret[index] = in[i];
		}
	}
	return ret;
}

__device__ __host__ bool isZero(const float x)
{
	if(x>0)
		return true;
	else
		return false;
}

float * thrust_tream_compact(float * in, int n)
{
	float * tmp = new float[n];

	thrust::copy_if(in,in+n,tmp,isZero);


	return tmp;
}

int main()
{
	double cpu_start;
	double cpu_stop;

	float * out1 = new float[NUM];
	float * out2 = new float[NUM];
	float * out3 = new float[NUM];
	float * out4 = new float[NUM];
	float * in = new float[NUM];

	for( int i = 0; i < NUM; i++)
	{
		in[i] = 2;
	}

	//1: serial prefix sum
	cpu_start = clock();
	CPU_prefix_sum(in, out1, NUM);
	cpu_stop = clock();
	printf ("#CPU prefix sum performance:  %f ms.\n",1000.0* (double)((cpu_stop-cpu_start)/(double)CLOCKS_PER_SEC));

	//for(int i = 0; i < NUM ; ++i)
	//cout << out1[NUM-1] <<  endl;
	
	//2: naive parallel prefix sum

	GPU_naive_prefix_sum(in, out2, NUM);

	//for(int i = 0; i < NUM ; ++i)
	//cout << out2[NUM-1]  << endl;

	//3a: parallel prefix sum with shared memory for a single block
	//single_shared_prefix_sum(in, out3,  SINGLE_SHARED_NUM);
	//for(int i = 0; i < SINGLE_SHARED_NUM ; ++i)
	//cout << out3[SINGLE_SHARED_NUM-1] << endl;

	//3b: parallel prefix sum with shared memory for arbitary array length

	GPU_shared_prefix_sum(in,out4,NUM);


	//for(int i = 0; i < NUM ; ++i)
	//cout << out4[NUM-1] << endl;
	
	//test CPU stream compact
	cpu_start = clock();
	float * out5 = CPU_stream_compact( in, NUM);
	cpu_stop = clock();
	printf ("#CPU stream compact performance:  %f ms.\n",1000.0*(double)((cpu_stop-cpu_start)/(double)CLOCKS_PER_SEC));
	/*int i = 0;
	while(out5[i]>0)
	{
		cout << out5[i] << endl;
		i++;
	}*/

	// test GPU stream compact
	float * out6 = new float[NUM];
	GPU_stream_compact(in, out6,NUM);

	// test thrust version
	cpu_start = clock();
	float * out7 = thrust_tream_compact(in, NUM);
	cpu_stop = clock();
	printf ("#Thrust stream compact performance:  %f ms.\n",1000.0*(double)((cpu_stop-cpu_start)/(double)CLOCKS_PER_SEC));


	free(out1);
	free(out2);
	free(out3);
	free(out4);
	free(out5);
	free(out6);
	free(out7);
	free(in);
	int c;
	cin >> c;
}