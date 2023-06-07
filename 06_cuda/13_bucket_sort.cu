#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init_bucket(int *bucket) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bucket[i] = 0;
}

__global__ void add(int *key, int *bucket) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&bucket[key[i]], 1);
}

__global__ void scan( int *bucket,int *a, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    // prefix sum     
    for(int j=1; j<range; j<<=1){
        a[i] = bucket[i];
        __syncthreads();
        bucket[i] += a[i-j];
        __syncthreads();
    }
}

__global__ void assignment(int *key, int *bucket, int n, int range){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j=0; j<range; j++){
        if(i < bucket[j] && i >= bucket[j-1]){
            key[i] = j;
            return;
        }
    }
}

int main() {
    int n = 50;
    int range = 5;
    int *key;
    cudaMallocManaged(&key, n*sizeof(int));

    for (int i=0; i<n; i++) {
        key[i] = rand() % range;
        printf("%d ",key[i]);
    }
    printf("\n");

    int *bucket;
    cudaMallocManaged(&bucket, range*sizeof(int));
    initialize<<<1, range>>>(bucket);
    cudaDeviceSynchronize();
   
    reduction<<<1, n>>>(bucket, key);
    cudaDeviceSynchronize();
  
    int *tmp;
    cudaMallocManaged(&tmp, range*sizeof(int));
    makeoffset<<<1, range>>>(bucket, tmp, range);
    cudaDeviceSynchronize();

    sort<<<1, n>>>(bucket, key, range);
    cudaDeviceSynchronize();

    for (int i=0; i<n; i++) {
        printf("%d ",key[i]);
    }
    printf("\n");

    cudaFree(bucket);
    cudaFree(key);
}
