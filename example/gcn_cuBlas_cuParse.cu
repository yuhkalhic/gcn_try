#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;
int *d_raw_graph;
float *d_edge_val;
int *d_edge_index, *d_degree;

void readGraph(char *fname)
{
    ifstream infile(fname);

    int source;
    int end;

    infile >> v_num >> e_num;

    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF)
            break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

__global__ void raw_graph_to_AdjacencyList_kernel(int *raw_graph, int *edge_index, float *edge_val, int *degree, int e_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < e_num)
    {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        atomicAdd(&degree[src], 1);
        edge_index[dst * e_num + src] = 1;
    }
}

__global__ void edgeNormalization_kernel(int *edge_index, float *edge_val, int *degree, int v_num, int e_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < v_num)
    {
        for (int j = 0; j < e_num; j++)
        {
            if (edge_index[i * e_num + j] == 1)
            {
                float val = 1 / sqrtf(degree[i]) / sqrtf(degree[j]);
                edge_val[i * e_num + j] = val;
            }
        }
    }
}

void raw_graph_to_AdjacencyList()
{
    cudaMalloc(&d_edge_index, v_num * e_num * sizeof(int));
    cudaMalloc(&d_edge_val, v_num * e_num * sizeof(float));
    cudaMalloc(&d_degree, v_num * sizeof(int));
    cudaMemset(d_edge_index, 0, v_num * e_num * sizeof(int));
    cudaMemset(d_edge_val, 0, v_num * e_num * sizeof(float));
    cudaMemset(d_degree, 0, v_num * sizeof(int));

    cudaMalloc(&d_raw_graph, raw_graph.size() * sizeof(int));
    cudaMemcpy(d_raw_graph, raw_graph.data(), raw_graph.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (e_num + blockSize - 1) / blockSize;
    raw_graph_to_AdjacencyList_kernel<<<numBlocks, blockSize>>>(d_raw_graph, d_edge_index, d_edge_val, d_degree, e_num);
}

void edgeNormalization()
{
    int blockSize = 256;
    int numBlocks = (v_num + blockSize - 1) / blockSize;
    edgeNormalization_kernel<<<numBlocks, blockSize>>>(d_edge_index, d_edge_val, d_degree, v_num, e_num);
}

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    memset(dst, 0, num * sizeof(float));
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
    float *d_in_X, *d_out_X, *d_W;
    cudaMalloc(&d_in_X, v_num * in_dim * sizeof(float));
    cudaMalloc(&d_out_X, v_num * out_dim * sizeof(float));
    cudaMalloc(&d_W, in_dim * out_dim * sizeof(float));

    cudaMemcpy(d_in_X, in_X, v_num * in_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, in_dim * out_dim * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_dim, v_num, in_dim, &alpha, d_W, out_dim, d_in_X, in_dim, &beta, d_out_X, out_dim);

    cudaMemcpy(out_X, d_out_X, v_num * out_dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in_X);
    cudaFree(d_out_X);
    cudaFree(d_W);
    cublasDestroy(handle);
}

__global__ void AX_kernel(float *in_X, float *out_X, int *edges, float *vals, int *edge_counts, int dim, int v_num)
{
    extern __shared__ float shared_mem[];
    float *shared_X = shared_mem;
    float *shared_vals = shared_mem + blockDim.x * dim;

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < v_num)
    {
        int edge_start = row == 0 ? 0 : edge_counts[row - 1];
        int edge_end = edge_counts[row];

        for (int d = 0; d < dim; d++)
        {
            shared_X[threadIdx.x * dim + d] = in_X[row * dim + d];
        }
        for (int edge = edge_start; edge < edge_end; edge++)
        {
            int nbr = edges[edge];
            shared_vals[threadIdx.x] = vals[edge];
            __syncthreads();

            float weight = shared_vals[threadIdx.x];
            for (int d = 0; d < dim; d++)
            {
                atomicAdd(&out_X[row * dim + d], in_X[nbr * dim + d] * weight);
            }
            __syncthreads();
        }
    }
}

void AX(int dim, float *in_X, float *out_X)
{
    int *d_edges;
    float *d_vals, *d_in_X, *d_out_X;
    int *d_edge_counts;

    cudaMalloc(&d_edges, e_num * sizeof(int));
    cudaMalloc(&d_vals, e_num * sizeof(float));
    cudaMalloc(&d_in_X, v_num * dim * sizeof(float));
    cudaMalloc(&d_out_X, v_num * dim * sizeof(float));
    cudaMalloc(&d_edge_counts, v_num * sizeof(int));

    cudaMemcpy(d_edges, d_edge_index, e_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, d_edge_val, e_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_X, in_X, v_num * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_X, out_X, v_num * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_edge_counts, d_degree, v_num * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks((v_num + 255) / 256);
    size_t shared_memory_size = threads.x * dim * sizeof(float) + threads.x * sizeof(float);
    AX_kernel<<<blocks, threads, shared_memory_size>>>(d_in_X, d_out_X, d_edges, d_vals, d_edge_counts, dim, v_num);

    cudaMemcpy(out_X, d_out_X, v_num * dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_edges);
    cudaFree(d_vals);
    cudaFree(d_in_X);
    cudaFree(d_out_X);
    cudaFree(d_edge_counts);
}

__global__ void ReLU_kernel(float *X, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        X[idx] = max(0.0f, X[idx]);
    }
}

void ReLU(int dim, float *X)
{
    float *d_X;
    cudaMalloc(&d_X, v_num * dim * sizeof(float));
    cudaMemcpy(d_X, X, v_num * dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks((v_num * dim + 255) / 256);
    dim3 threads(256);
    ReLU_kernel<<<blocks, threads>>>(d_X, v_num * dim);

    cudaMemcpy(X, d_X, v_num * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_X);
}

__global__ void LogSoftmax_kernel(float *X, int v_num, int dim)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < v_num)
    {
        float *row = &X[idx * dim];
        float max_val = *row;
        for (int i = 1; i < dim; ++i)
        {
            max_val = max(max_val, row[i]);
        }
        float sum = 0;
        for (int i = 0; i < dim; ++i)
        {
            sum += exp(row[i] - max_val);
        }
        sum = log(sum);
        for (int i = 0; i < dim; ++i)
        {
            row[i] = row[i] - max_val - sum;
        }
    }
}

void LogSoftmax(int dim, float *X)
{
    float *d_X;
    cudaMalloc(&d_X, v_num * dim * sizeof(float));
    cudaMemcpy(d_X, X, v_num * dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks((v_num + 255) / 256);
    dim3 threads(256);
    LogSoftmax_kernel<<<blocks, threads>>>(d_X, v_num, dim);

    cudaMemcpy(X, d_X, v_num * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_X);
}

float MaxRowSum(float *X, int dim)
{
    float(*tmp_X)[dim] = (float(*)[dim])X;
    float max = -__FLT_MAX__;

    for (int i = 0; i < v_num; i++)
    {
        float sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += tmp_X[i][j];
        }
        if (sum > max)
            max = sum;
    }
    return max;
}

void freeFloats()
{
    free(X0);
    free(W1);
    free(W2);
    free(X1);
    free(X2);
    free(X1_inter);
    free(X2_inter);
    cudaFree(d_raw_graph);
    cudaFree(d_edge_val);
    cudaFree(d_edge_index);
    cudaFree(d_degree);
}

void somePreprocessing()
{
    raw_graph_to_AdjacencyList();
}

int main(int argc, char **argv)
{
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);
    F2 = atoi(argv[3]);

    readGraph(argv[4]);
    readFloat(argv[5], X0, v_num * F0);
    readFloat(argv[6], W1, F0 * F1);
    readFloat(argv[7], W2, F1 * F2);

    initFloat(X1, v_num * F1);
    initFloat(X1_inter, v_num * F1);
    initFloat(X2, v_num * F2);
    initFloat(X2_inter, v_num * F2);

    TimePoint start = chrono::steady_clock::now();

    TimePoint prepross_start = chrono::steady_clock::now();
    somePreprocessing();
    TimePoint prepross_end = chrono::steady_clock::now();
    chrono::duration<double> prepross_ = prepross_end - prepross_start;
    double prepross_time = prepross_.count() * 1e3;
    printf("prepross_time: %.8lf\n", prepross_time);

    TimePoint edgeNorm_start = chrono::steady_clock::now();
    edgeNormalization();
    TimePoint edgeNorm_end = chrono::steady_clock::now();
    chrono::duration<double> edgeNorm_ = edgeNorm_end - edgeNorm_start;
    double edgeNorm_time = edgeNorm_.count() * 1e3;
    printf("edgeNorm_time: %.8lf\n", edgeNorm_time);

    TimePoint XW1_start = chrono::steady_clock::now();
    XW(F0, F1, X0, X1_inter, W1);
    TimePoint XW1_end = chrono::steady_clock::now();
    chrono::duration<double> XW1_ = XW1_end - XW1_start;
    double XW1_time = XW1_.count() * 1e3;
    printf("XW1_time: %.8lf\n", XW1_time);

    TimePoint AX1_start = chrono::steady_clock::now();
    AX(F1, X1_inter, X1);
    TimePoint AX1_end = chrono::steady_clock::now();
    chrono::duration<double> AX1_ = AX1_end - AX1_start;
    double AX1_time = AX1_.count() * 1e3;
    printf("AX1_time: %.8lf\n", AX1_time);

    TimePoint ReLU_start = chrono::steady_clock::now();
    ReLU(F1, X1);
    TimePoint ReLU_end = chrono::steady_clock::now();
    chrono::duration<double> ReLU_ = ReLU_end - ReLU_start;
    double ReLU_time = ReLU_.count() * 1e3;
    printf("ReLU_time: %.8lf\n", ReLU_time);

    TimePoint XW2_start = chrono::steady_clock::now();
    XW(F1, F2, X1, X2_inter, W2);
    TimePoint XW2_end = chrono::steady_clock::now();
    chrono::duration<double> XW2_ = XW2_end - XW2_start;
    double XW2_time = XW2_.count() * 1e3;
    printf("XW2_time: %.8lf\n", XW2_time);

    TimePoint AX2_start = chrono::steady_clock::now();
    AX(F2, X2_inter, X2);
    TimePoint AX2_end = chrono::steady_clock::now();
    chrono::duration<double> AX2_ = AX2_end - AX2_start;
    double AX2_time = AX2_.count() * 1e3;
    printf("AX2_time: %.8lf\n", AX2_time);

    TimePoint LogSoftmax_start = chrono::steady_clock::now();
    LogSoftmax(F2, X2);
    TimePoint LogSoftmax_end = chrono::steady_clock::now();
    chrono::duration<double> LogSoftmax_ = LogSoftmax_end - LogSoftmax_start;
    double LogSoftmax_time = LogSoftmax_.count() * 1e3;
    printf("LogSoftmax_time: %.8lf\n", LogSoftmax_time);

    TimePoint max_sum_start = chrono::steady_clock::now();
    float max_sum = MaxRowSum(X2, F2);
    TimePoint max_sum_end = chrono::steady_clock::now();
    chrono::duration<double> max_sum_ = max_sum_end - max_sum_start;
    double max_sum_time = max_sum_.count() * 1e3;
    printf("max_sum_time: %.8lf\n", max_sum_time);

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    printf("%.8f\n", max_sum);
    printf("total time: %.8lf\n\n", l_timeMs);

    freeFloats();
}
