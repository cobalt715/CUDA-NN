#ifdef COBALT_715_USE_CUDA

#include <stdexcept>
#include "vec.cuh"
#include "op.cuh"
#include "nn/cuda/config.cuh"

namespace cobalt_715::nn::ops::vec::cuda{

template<typename Op>
__global__ void binary_op(const float *a,const float *b,float *out,const int64_t n){
  const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if(x >= n) return;

  out[x] = Op::f(a[x],b[x]);
}

void add(const float *a,const float *b,float *out,const int64_t n){
  binary_op<op::add><<<100,100>>>(a,b,out,n);
}

}//namespace cobalt_715::nn::ops::vec::cuda

#endif //COBALT_715_USE_CUDA