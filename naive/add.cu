#include <cuda.h>
#include <stdio.h>
// linux版本
#include <sys/time.h>
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

void initCpu(float *hostA, float *hostB, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostA[i] = 1;
        hostB[i] = 1;
    }
}
void addCpu(float *hostA, float *hostB, float *hostC, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostC[i] = hostA[i] + hostB[i];
    }
}
__global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // 计算全局索引
    if (index < n)
    {
        deviceC[index] = deviceA[index] + deviceB[index];
    }
}
int main()
{
    // 初始化数组指针
    float *hostA, *hostB, *hostC, *serialC;
    int n = 102400;

    hostA = (float *)malloc(n * sizeof(float));
    hostB = (float *)malloc(n * sizeof(float));
    hostC = (float *)malloc(n * sizeof(float));
    serialC = (float *)malloc(n * sizeof(float));
    initCpu(hostA, hostB, n);
    double stC, elaC;
    stC = get_walltime();
    addCpu(hostA, hostB, serialC, n);
    elaC = get_walltime() - stC;
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    // 分配内存
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&dB, n * sizeof(float));
    cudaMalloc((void **)&dC, n * sizeof(float));
    // 内存复制
    cudaMemcpy(dA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, n * sizeof(float), cudaMemcpyHostToDevice);
    // 定义两个 CUDA 事件对象 start 和 stop，用于记录内核执行的开始和结束时间。
    cudaEvent_t start, stop;
    // 用于存储内核执行的时间
    float ker_time = 0;
    // CUDA 事件对象
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录内核执行的开始时间。0 表示使用默认的流（即当前流
    cudaEventRecord(start, 0);
    //  定义每个块的线程数为 1024。
    int BLOCK_DIM = 1024;
    // 计算所需的块数。
    int num_block_x = n / BLOCK_DIM;
    // 定义每个块在 y 方向上的线程数为 1
    int num_block_y = 1;
    // grid_dim 是一个 dim3 对象，表示网格在 x、y 和 z 方向上的尺寸。
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);
    addKernel<<<grid_dim, block_dim>>>(dA, dB, dC, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(hostC, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    ela = get_walltime() - st;
    printf("n = %d: \n CPU use time:%.4f\n GPU use time:%.4f\n kernel time:%.4f\n", n, elaC, ela, ker_time / 1000.0);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}