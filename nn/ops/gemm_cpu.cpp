#include <immintrin.h>
#include <cstdint>
#include "gemm.cuh"
#include "nn/dtype.hpp"

namespace cobalt_715::nn::ops::cpu{

//素朴実装
template<nn::mutable_dtype T>
void kernel(const T *a_pack,
            const T *bt_pack,
            T *out_pack,
            const int64_t IB,
            const int64_t JB,
            const int64_t KB){

  for(int64_t i = 0;i < IB;i++){
    for(int64_t j = 0;j < JB;j++){
      T sum = 0;
      for(int64_t k = 0;k < KB;k++){
        sum += a_pack[i * KB + k] * bt_pack[j * KB + k];
      }
      out_pack[i * JB + j] = sum;
    }
  }
}

//レジスタブロッキング
template<nn::mutable_dtype T>
void kernel_reg_4_4(const T *a_pack,
                    const T *bt_pack,
                    T *out_pack,
                    const int64_t IB,
                    const int64_t JB,
                    const int64_t KB){

  for(int64_t i = 0;i < IB;i += 4){
    const T *adptr0 = a_pack + i * KB;
    const T *adptr1 = a_pack + (i + 1) * KB;
    const T *adptr2 = a_pack + (i + 2) * KB;
    const T *adptr3 = a_pack + (i + 3) * KB;

    T *odptr0 = out_pack + i * JB;
    T *odptr1 = out_pack + (i + 1) * JB;
    T *odptr2 = out_pack + (i + 2) * JB;
    T *odptr3 = out_pack + (i + 3) * JB;

    for(int64_t j = 0;j < JB;j += 4){
      T o00=0,o01=0,o02=0,o03=0;
      T o10=0,o11=0,o12=0,o13=0;
      T o20=0,o21=0,o22=0,o23=0;
      T o30=0,o31=0,o32=0,o33=0;

      const T *btdptr0 = bt_pack + j * KB;
      const T *btdptr1 = bt_pack + (j + 1) * KB;
      const T *btdptr2 = bt_pack + (j + 2) * KB;
      const T *btdptr3 = bt_pack + (j + 3) * KB;

      T *optr0 = odptr0 + j;
      T *optr1 = odptr1 + j;
      T *optr2 = odptr2 + j;
      T *optr3 = odptr3 + j;

      for(int64_t k = 0;k < KB;k++){
        const float a0 = adptr0[k];
        const float a1 = adptr1[k];
        const float a2 = adptr2[k];
        const float a3 = adptr3[k];

        const float bt0 = btdptr0[k];
        const float bt1 = btdptr1[k];
        const float bt2 = btdptr2[k];
        const float bt3 = btdptr3[k];

        o00 += a0 * bt0;o01 += a0 * bt1;o02 += a0 * bt2;o03 += a0 * bt3;
        o10 += a1 * bt0;o11 += a1 * bt1;o12 += a1 * bt2;o13 += a1 * bt3;
        o20 += a2 * bt0;o21 += a2 * bt1;o22 += a2 * bt2;o23 += a2 * bt3;
        o30 += a3 * bt0;o31 += a3 * bt1;o32 += a3 * bt2;o33 += a3 * bt3;
      }

      optr0[0] = o00; optr0[1] = o01; optr0[2] = o02; optr0[3] = o03;
      optr1[0] = o10; optr1[1] = o11; optr1[2] = o12; optr1[3] = o13;
      optr2[0] = o20; optr2[1] = o21; optr2[2] = o22; optr2[3] = o23;
      optr3[0] = o30; optr3[1] = o31; optr3[2] = o32; optr3[3] = o33;
    }
  }
}

//パックする
//コピー先のアドレス、コピー元のアドレス、stride,コピーしたい位置、コピーしたいサイズ
//Transpose=trueなら転置
template<bool Transpose,nn::mutable_dtype T>
inline void set_pack(T *dst,
                     const T *src,
                     const int64_t rows,
                     const int64_t cols,
                     const int64_t row_stride,
                     const int64_t col_stride,
                     const int64_t row,
                     const int64_t col,
                     const int64_t row_size,
                     const int64_t col_size){

  for(int64_t r = 0;r < row_size;r++){
    for(int64_t c = 0;c < col_size;c++){
      //範囲外は0埋め
      if(r + row >= rows || c + col >= cols){
        if constexpr (!Transpose){
          dst[r * col_size + c] = 0;
        }else{
          dst[c * row_size + r] = 0;
        }
      }else{
        if constexpr (!Transpose){
          dst[r * col_size + c] = src[(r + row) * row_stride + (c + col) * col_stride];
        }else{
          dst[c * row_size + r] = src[(r + row) * row_stride + (c + col) * col_stride];
        }
      }
    }
  }
}

//outに代入
template<nn::mutable_dtype T>
inline void write_out(T *dst,
                      const T *src,
                      const T alpha,
                      const T beta,
                      const int64_t rows,
                      const int64_t cols,
                      const int64_t row_stride,
                      const int64_t col_stride,
                      const int64_t row,
                      const int64_t col,
                      const int64_t row_size,
                      const int64_t col_size){

  for(int64_t r = 0;r < row_size;r++){
    for(int64_t c = 0;c < col_size;c++){
      if(r + row >= rows || c + col >= cols) continue;
      T *od = &dst[(r + row) * row_stride + (c + col) * col_stride];
      *od = alpha * src[r * col_size + c] + beta * (*od);
    }
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

  //ブロックサイズ
  //kernelがそれを前提にしているため32の倍数のみにする
  constexpr int64_t IB = 12;
  constexpr int64_t JB = 12;
  constexpr int64_t KB = 12;

  alignas(64) T a_pack[IB * KB];
  alignas(64) T b_pack[KB * JB];
  alignas(64) T out_pack[IB * JB];

  for(int64_t ii = 0;ii < I;ii += IB){
    for(int64_t jj = 0;jj < J;jj += JB){
      for(int64_t kk = 0;kk < K;kk += KB){
        set_pack<false,T>(a_pack,a,I,K,a_row_stride,a_col_stride,ii,kk,IB,KB);
        //bは転置
        set_pack<true,T>(b_pack,b,K,J,b_row_stride,b_col_stride,kk,jj,KB,JB);

        //kernel<T>(a_pack,b_pack,out_pack,IB,JB,KB);
        kernel_reg_4_4<T>(a_pack,b_pack,out_pack,IB,JB,KB);
      }
      write_out<T>(out,out_pack,alpha,beta,I,J,out_row_stride,out_col_stride,ii,jj,IB,JB);
    }
  }
}

#define INSTANTIATE_GEMM_IMPL(T) \
  template void gemm_impl<T>(const T alpha,const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  const T beta,T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t I,const int64_t J,const int64_t K);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_GEMM_IMPL)

}//namespace cobalt_715::nn::ops::cpu