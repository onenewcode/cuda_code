#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <cub/block/block_reduce.cuh>
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
float addCpu(float *hostA, int n)
{
    float tmp = 0.0f; // 表示C++中的负无穷
    for (int i = 0; i < n; i++)
    {
        tmp += hostA[i];
    }
    return tmp;
}
// 使用模板可以让你在编译时指定块的大小
template <int BLOCK_DIM>
__global__ void addKernel(float *dA, int n, float *globalMax, int strategy)
{
    // 定义一个共享内存数组 tmpSum，每个块的线程都可以访问。
    __shared__ float tmpSum[BLOCK_DIM];
    float tmp = 0.0f;
    for (int id = threadIdx.x; id < n; id += BLOCK_DIM)
    {
        tmp += dA[id];
    }
    // 用于存储不同的grid_dim的初始值
    tmpSum[threadIdx.x] = tmp;
    // 同步代码块，确保所有线程完成对tmpSum的写入操作后，再继续执行后续代码。
    __syncthreads();
    if (strategy == 0)
    {
        //  简单规约
        for (int step = 1; step < BLOCK_DIM; step *= 2)
        {
            // 
            if (threadIdx.x % (2 * step) == 0)
            {
                tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            globalMax[0] = tmpSum[0];
        }
    }
    else if (strategy == 1)
    {
        for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
        {
            if (threadIdx.x < step)
            {
                tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            globalMax[0] = tmpSum[0];
        }
    }
    else if (strategy == 2)
    {
        // __shared__ float val[32]; 定义一个共享内存数组val，大小为32，用于存储中间结果。
        __shared__ float val[32];
        float data = tmpSum[threadIdx.x];
        // 0xffffffff 是一个32位的整数，其二进制表示为 1111 1111 1111 1111 1111 1111 1111 1111，也就是说，它的所有32位都是1。当这个值作为掩码传递给 __shfl_down_sync() 函数时，意味着warp中的所有32个线程都参与该操作。
        // 1下移16位的值相加。这意味着，如果一个线程的索引是0，它会获取索引16的线程的data值；如果索引是1，它会获取索引17的线程的data值，以此类推
        data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
        data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
        data += __shfl_down_sync(0xffffffff, data, 4);
        data += __shfl_down_sync(0xffffffff, data, 2);
        data += __shfl_down_sync(0xffffffff, data, 1);
        if (threadIdx.x % 32 == 0)
        {
            val[threadIdx.x / 32] = data;
        }
        __syncthreads();
        if (threadIdx.x < 32)
        {
            data = val[threadIdx.x];
            data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
            data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
            data += __shfl_down_sync(0xffffffff, data, 4);
            data += __shfl_down_sync(0xffffffff, data, 2);
            data += __shfl_down_sync(0xffffffff, data, 1);
        }

        __syncthreads();
        if (threadIdx.x == 0)
        {
            globalMax[0] = data;
        }
    }
    else
    {
        // 块级别的归约操作
        typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce; //<float,..>里面的float表示返回值的类型
        //  定义一个共享内存数组,用于存储归约过程中的临时数据。
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float block_sum = BlockReduce(temp_storage).Reduce(tmpSum[threadIdx.x], cub::Sum());
        if (threadIdx.x == 0)
        {
            globalMax[0] = block_sum;
        }
    }
}
int main()
{
    float *hostA;
    int n = 1024;
    int strategy = 2;
    int repeat = 100;
    hostA = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        hostA[i] = (i % 10) * 1e-1;
    }
    float hostMax;
    double st, ela;
    st = get_walltime();

    float *dA, *globalMax;
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&globalMax, sizeof(float));
    cudaMemcpy(dA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int BLOCK_DIM = 64;
    int num_block_x = (n+BLOCK_DIM-1) / BLOCK_DIM;
    int num_block_y = 1;
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);
    for (int i = 0; i < repeat; i++)
    {
        // <1024>: 模板参数，指定每个块的线程数为 1024
        addKernel<64><<<grid_dim, block_dim>>>(dA, n, globalMax, strategy);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(&hostMax, globalMax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(globalMax);
    ela = 1000 * (get_walltime() - st);
    printf("n = %d: strategy:%d, GPU use time:%.4f ms, kernel time:%.4f ms\n", n, strategy, ela, ker_time / repeat);
    printf("CPU sum:%.2f, GPU sum:%.2f\n", addCpu(hostA, n), hostMax);
    free(hostA);

    return 0;
}
