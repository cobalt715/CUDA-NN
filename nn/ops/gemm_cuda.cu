#ifdef COBALT_715_USE_CUDA

#include <cstdint>
#include "gemm.cuh"
#include "nn/dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::ops::cuda{

//OUT = AB
//16*16*K
template<nn::mutable_dtype T>
__global__ void cuda_gemm(const T alpha,

                          const T *a,
                          const int64_t a_row_stride,
                          const int64_t a_col_stride,

                          const T *b,
                          const int64_t b_row_stride,
                          const int64_t b_col_stride,

                          const T beta,

                          T *out,
                          const int64_t out_row_stride,
                          const int64_t out_col_stride,

                          const int64_t I,
                          const int64_t J,
                          const int64_t K){

  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;



  /*if(row >= I || col >= J) return;

  T sum = 0;

  for(int64_t k = 0;k < K;k++){
    sum += a[row * a_row_stride + k * a_col_stride] * b[k * b_row_stride + col * b_col_stride];
  }

  T *od = &out[row * out_row_stride + col * out_col_stride];
  *od = (alpha * sum) + (beta * (*od));*/



  const int64_t tx = threadIdx.x;
  const int64_t ty = threadIdx.y;

  __shared__ T a_pack[16][16];
  __shared__ T bt_pack[16][16];

  T sum = 0;

  for(int64_t kk = 0;kk < K;kk += 16){
    if(row < I && kk + tx < K){
      a_pack[ty][tx] =
        a[row * a_row_stride + (kk + tx) * a_col_stride];
    }else{
      a_pack[ty][tx] = 0.0f;
    }

    if(col < J && kk + ty < K){
      bt_pack[tx][ty] =
        b[(kk + ty) * b_row_stride + col * b_col_stride];
    }else{
      bt_pack[tx][ty] = 0.0f;
    }

    __syncthreads();

    for(int64_t k = 0;k < 16;k++){
      sum += a_pack[ty][k] * bt_pack[tx][k];
    }

    __syncthreads();
  }

  if(row < I && col < J){
    T *od = &out[row * out_row_stride + col * out_col_stride];
    *od = (alpha * sum) + (beta * (*od));
  }
}

template<nn::mutable_dtype T>
void gemm_impl(const T alpha,

               const T *a,
               const int64_t a_row_stride,
               const int64_t a_col_stride,

               const T *b,
               const int64_t b_row_stride,
               const int64_t b_col_stride,

               const T beta,

               T *out,
               const int64_t out_row_stride,
               const int64_t out_col_stride,

               const int64_t I,
               const int64_t J,
               const int64_t K){

  const dim3 grid((J + 15) / 16,(I + 15) / 16);
  const dim3 block(16,16);

  cuda_gemm<T><<<grid,block>>>(alpha,
                               a,a_row_stride,a_col_stride,
                               b,b_row_stride,b_col_stride,
                               beta,
                               out,out_row_stride,out_col_stride,
                               I,J,K);
  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

#define INSTANTIATE_GEMM_IMPL(T) \
  template void gemm_impl<T>(const T alpha,const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  const T beta,T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t I,const int64_t J,const int64_t K);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_GEMM_IMPL)

}//namespace cobalt_715::nn::ops::cuda

#endif