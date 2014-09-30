# PART 2: NAIVE PREFIX SUM
As we can see from the graph, GPU naive scan is faster than CPU version when the length of the array is smaller than 1.2 million. And the time consumption for GPU naive version become worse when the array size increases.The reason for naive version become inefficient when array size is too big is because there are too many global memory access. 

# PART 3 : OPTIMIZING PREFIX SUM

As can see from the graph, GPU shared memory (green) is always faster than both GPU naive version (red) and CPU version (blue). In the experiment, I used 1024 block size. If block size is smaller, the GPU shared memory version can run much faster. The main reason for shared memory become not that efficient after 1 million is because the bank conflict starts to happen more frequently. So, some threads have to wait until the shared memory is ready to use. 

# PART 4 : ADDING SCATTER

According to the graph, the thrust version is the most inefficient version, and the GPU version is the most efficient version. The thrust version is that slow may due to the way I calculated clock time is in term of CPU instead of GPU. If thrust can handle the bank conflits well, then it should run faster than my GPU version. 
