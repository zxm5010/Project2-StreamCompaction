#include <iostream>
#include <stdlib.h>
#include "StreamCompaction.h"
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

const int NUM = 12000000;
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

void CPU_stream_compact(float * in, float * out, int n)
{
	int compact_size = 0;
	CPU_scatter(in,out,n);
	compact_size = out[n-1] + 1;
	
	float * ret = new float[compact_size];


	for(int i = 1; i < n; ++i)
	{
		if(out[i] != out[i-1])
		{
			int index = out[i]-1;
			//cout << index << endl;
			out[index] = in[i-1];
		}
	}
}

int main()
{
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
	CPU_prefix_sum(in, out1, NUM);
	//for(int i = 0; i < NUM ; ++i)
	cout << out1[NUM-1] <<  endl;
	
	//2: naive parallel prefix sum
	GPU_naive_prefix_sum(in, out2, NUM);
	//for(int i = 0; i < NUM ; ++i)
	cout << out2[NUM-1]  << endl;

	//3a: parallel prefix sum with shared memory for a single block
	//single_shared_prefix_sum(in, out3,  SINGLE_SHARED_NUM);
	//for(int i = 0; i < SINGLE_SHARED_NUM ; ++i)
	//cout << out3[SINGLE_SHARED_NUM-1] << endl;

	//3b: parallel prefix sum with shared memory for arbitary array length
	GPU_shared_prefix_sum(in,out4,NUM);
	//for(int i = 0; i < NUM ; ++i)
	cout << out4[NUM-1] << endl;
	
	//test CPU stream compact
	float in2[] = {0,0,3,4,0,6,6,7,0,1};
	float * out5 = new float[10];
	
	CPU_stream_compact( in2,  out5, 10);

	int i = 0;
	while(out5[i]!=NULL)
	{
		cout << out5[i] << endl;
		++i;
	}


	free(out1);
	free(out2);
	free(out3);
	free(out4);
	free(in);
	int c;
	cin >> c;
}