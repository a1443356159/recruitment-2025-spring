//没有修改数组顺序
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#include "utils.h"
#include "omp.h"

void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V, V_temp=(V_tensor_t)malloc(sizeof(float)*ti.tile_in_h*ti.tile_in_w*collapsed_dim_size);

//存在内存访问不连续的问题
  float A[6][6]={
    {4,0,0,0,0,0},
    {0,-4,4,-2,2,4},
    {-5,-4,-4,-1,-1,0},
    {0,1,-1,2,-2,-5},
    {1,1,1,1,1,0},
    {0,0,0,0,0,1}
   }
  ,B[6][6]={
    {4,0,0,0,0,0},
    {0,-4,4,-2,2,4},
    {-5,-4,-4,-1,-1,0},
    {0,1,-1,2,-2,-5},
    {1,1,1,1,1,0},
    {0,0,0,0,0,1}
  };
  assert(ti.tile_in_w==6&&ti.tile_in_h==6);
  // 第一步：水平变换（使用矩阵 A）
#pragma omp parallel for collapse(2)
  for (int j = 0; j < 6; ++j) {
    for (int k = 0; k < 6; ++k) {
      for (int64_t idx = 0; idx < collapsed_dim_size; ++idx) {
        V_temp[j][k][idx] = A[0][j] * packed_image_tensor[0][k][idx]+
                            A[1][j] * packed_image_tensor[1][k][idx]+
                            A[2][j] * packed_image_tensor[2][k][idx]+
                            A[3][j] * packed_image_tensor[3][k][idx]+
                            A[4][j] * packed_image_tensor[4][k][idx]+
                            A[5][j] * packed_image_tensor[5][k][idx];
      }
    }
  }

  // 第二步：垂直变换（使用矩阵 B）
#pragma omp parallel for collapse(2)
  for (int h = 0; h < 6; ++h) {
    for (int k = 0; k < 6; ++k) {
      for (int64_t idx = 0; idx < collapsed_dim_size; ++idx) {
       V_tensor[h][k][idx] = B[0][k] * V_temp[h][0][idx]+
                             B[1][k] * V_temp[h][1][idx]+
                             B[2][k] * V_temp[h][2][idx]+
                             B[3][k] * V_temp[h][3][idx]+
                             B[4][k] * V_temp[h][4][idx]+
                             B[5][k] * V_temp[h][5][idx];
      }
    }
  }
  free(V_temp);
}

void filter_transform(float *__restrict__ packed_filter,
                       float *__restrict__ U,
                       const filter_shape_t fs,
                       const U_shape_t us,
                       const int64_t collapsed_dim_size) {
  // 确保输入滤波器宽度为3（Winograd F(6,3)）
  assert(fs.w == 3 && "Filter width must be 3 for this transform");
  assert(us.h >= 6 && us.w >= 6 && "U shape must be at least 6x6");

  // 定义变换矩阵（6x3）
  const float G[6][3] = {
    { 1.0f/4.0f,  0.0f,       0.0f      },  // z0 = (1/4)f0
    {-1.0f/6.0f, -1.0f/6.0f, -1.0f/6.0f},  // z1 = (-1/6)(f0 + f1 + f2)
    {-1.0f/6.0f,  1.0f/6.0f, -1.0f/6.0f},  // z2 = (-1/6)(f0 - f1 + f2)
    { 1.0f/24.0f, 1.0f/12.0f, 1.0f/6.0f },  // z3 = (1/24)f0 + (1/12)f1 + (1/6)f2
    { 1.0f/24.0f, -1.0f/12.0f,1.0f/6.0f },  // z4 = (1/24)f0 - (1/12)f1 + (1/6)f2
    { 0.0f,       0.0f,       1.0f      }   // z5 = f2
  };

  // 类型定义（明确内存布局）
  typedef float (*packed_filter_tensor_t)[3][collapsed_dim_size]; // [c][w][idx]
  typedef float (*U_tensor_t)[6][collapsed_dim_size];           // [h][w][idx]

  // 指针转换
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  // 临时缓冲区（用于中间结果）
  float (*U_temp)[3][collapsed_dim_size] = (float (*)[3][collapsed_dim_size])malloc(
    sizeof(float) * 6 * 3 * collapsed_dim_size
  );

  // 第一步：w方向变换（矩阵乘法 G * F^T）
  #pragma omp parallel for collapse(2)
    for (int h = 0; h < 6; ++h)
      for (int w = 0; w < 3; ++w)
        for (int64_t idx = 0; idx < collapsed_dim_size; ++idx)
          U_temp[h][w][idx] = G[h][0] * packed_filter_tensor[0][w][idx]
                           +G[h][1] * packed_filter_tensor[1][w][idx]
                           +G[h][2] * packed_filter_tensor[2][w][idx];

  #pragma omp parallel for collapse(2)
    for (int h_out = 0; h_out < 6; ++h_out)
      for (int w_out = 0; w_out < 6; ++w_out)
        for (int64_t idx = 0; idx < collapsed_dim_size; ++idx)
          U_tensor[h_out][w_out][idx]=U_temp[h_out][0][idx] * G[w_out][0]
                                    +U_temp[h_out][1][idx] * G[w_out][1]
                                    +U_temp[h_out][2][idx] * G[w_out][2];
  free(U_temp);
}

void output_transform(float *__restrict__ M,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {
  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;

  // 临时变量维度: [6][4][collapsed_dim_size]
  typedef float(*Y_temp_t)[4][collapsed_dim_size];
  Y_temp_t Y_temp = (Y_temp_t)malloc(6 * 4 * collapsed_dim_size * sizeof(float));

  // 水平变换矩阵 C
  const float C[4][6] = {
    {1, 1, 1, 1, 1, 0},
    {0, 1, -1, 2, -2, 0},
    {0, 1, 1, 4, 4, 0},
    {0, 1, -1, 8, -8, 1}
  };

  // 垂直变换矩阵 D（与 C 相同）
  const float D[4][6] = {
    {1, 1, 1, 1, 1, 0},
    {0, 1, -1, 2, -2, 0},
    {0, 1, 1, 4, 4, 0},
    {0, 1, -1, 8, -8, 1}
  };

  assert(ti.tile_in_w == 6 && ti.tile_out_h == 4);

#pragma omp parallel for collapse(2)
    for (int h = 0; h < 6; ++h)
      for (int k = 0; k < 4; ++k)
        for (int64_t idx = 0; idx < collapsed_dim_size; idx++)
          Y_temp[h][k][idx]= C[k][0] * M_tensor[h][0][idx]
                + C[k][1] * M_tensor[h][1][idx]
                + C[k][2] * M_tensor[h][2][idx]
                + C[k][3] * M_tensor[h][3][idx]
                + C[k][4] * M_tensor[h][4][idx]
                + C[k][5] * M_tensor[h][5][idx];

#pragma omp parallel for collapse(2)
    for (int k = 0; k < 4; ++k)
      for (int w = 0; w < 4; ++w)
        for (int64_t idx = 0; idx < collapsed_dim_size; idx++)
          Y_tensor[k][w][idx] = D[k][0] * Y_temp[0][w][idx]
                 + D[k][1] * Y_temp[1][w][idx]
                 + D[k][2] * Y_temp[2][w][idx]
                 + D[k][3] * Y_temp[3][w][idx]
                 + D[k][4] * Y_temp[4][w][idx]
                 + D[k][5] * Y_temp[5][w][idx];
    free(Y_temp);
  }

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
#pragma omp parallel for collapse(2)
  for (int64_t h = 0; h < fs.h; ++h)
    for (int64_t w = 0; w < fs.w; ++w)
      for (int64_t oc = 0; oc < fs.oc; oc++)
        for (int64_t ic = 0; ic < fs.ic; ic++)
          packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
}

void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
  typedef float(*packedImage_tensor_t)[ti.tile_in_w][ti.num_tiles][is.ic];
  typedef float(*image_tensor_t)[is.ic][is.h][is.w];
  packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  image_tensor_t image_tensor = (image_tensor_t)image;
#pragma omp parallel for collapse(2)
  for (int64_t h = 0; h < ti.tile_in_h; ++h)
    for (int64_t w = 0; w < ti.tile_in_w; ++w)
      for (int64_t tile = 0; tile < ti.num_tiles; tile++)
        for (int64_t ic = 0; ic < is.ic; ic++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if (hh * 4 + h < is.h && ww * 4 + w < is.w)
            packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][(hh * 4 + h)][(ww * 4 + w)];
          else
            packed_image_tensor[h][w][tile][ic] = 0;
        }
}

void output_unpacking_store(float *__restrict__ Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) {
  typedef float(*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;
#pragma omp parallel for collapse(2)
  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    for (int64_t w = 0; w < ti.tile_out_w; ++w) {
      for (int64_t oc = 0; oc < os.oc; oc++) {
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if (hh * 4 + h < os.h && ww * 4 + w < os.w)
            out_tensor[batch][oc][(hh * 4 + h)][(ww * 4 + w)] = Y_tensor[h][w][oc][tile];
        }
      }
    }
  }
}
// 全局 cuBLAS 句柄
cublasHandle_t handle;

// 初始化 cuBLAS 句柄
void init_cublas() {
    cublasCreate(&handle);
}

// 销毁 cuBLAS 句柄
void destroy_cublas() {
    cublasDestroy(handle);
}

// 使用 cuBLAS 进行矩阵乘法
void sgemm_cublas(const int64_t M, const int64_t N, const int64_t K,
                  float *d_A, float *d_B, float *d_C) {
    float alpha = 1.0f;  // 缩放因子
    float beta = 0.0f;   // 缩放因子

    // 调用 cuBLAS 的 Sgemm 函数
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
}

void winograd_convolution(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
    /* new vars of shape */
    const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
    const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
    const out_shape_t os = get_output_shape(is, fs);
    const tiling_info_t ti = get_tiling_info(is, os);
    const U_shape_t us = get_U_shape(fs, ti);
    const V_shape_t vs = get_V_shape(is, ti);

    // 初始化 cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配 CPU 内存
    float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
    memset(packed_filter,0,sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
    float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
    memset(packed_image,0,sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
    float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
    memset(U,0,sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
    float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
    memset(V,0,sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
    float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
    memset(M,0,sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
    float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);
    memset(Y,0,sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);
    // 分配 GPU 内存
    float *d_U, *d_V, *d_M;
    cudaMalloc((void**)&d_U, sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
    cudaMalloc((void**)&d_V, sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
    cudaMalloc((void**)&d_M, sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);

    // 转换操作（在 CPU 上）
    filter_packing(filter, packed_filter, fs);
    filter_transform(packed_filter, U, fs, us, us.oc * us.ic);
    image_packing(image, packed_image, is, ti);
    image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

    // 数据拷贝到 GPU
    cudaMemcpy(d_U, U, sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
      for (int64_t w = 0; w < ti.tile_in_w; ++w) {
        int64_t U_offset = (h * ti.tile_in_w + w) * us.oc * us.ic;
        int64_t V_offset = (h * ti.tile_in_w + w) * vs.num_tiles * vs.ic;
        int64_t M_offset = (h * ti.tile_in_w + w) * us.oc * vs.num_tiles;

        cublasSgemm(
            handle,
            CUBLAS_OP_T,       // V 需要转置
            CUBLAS_OP_N,       // U 不转置
            vs.num_tiles,      // 输出列数（对应原代码的 vs.num_tiles）
            us.oc,            // 输出行数（对应原代码的 us.oc）
            us.ic,            // 公共维度（对应原代码的 us.ic）
            &alpha,
            d_V + V_offset, vs.ic,      // V 的 leading dimension 是原行数 vs.ic
            d_U + U_offset, us.ic,     // U 的 leading dimension 是原行数 us.ic
            &beta,
            d_M + M_offset, vs.num_tiles // M 的 leading dimension 是输出列数 vs.num_tiles
        );
      }
    }


    cudaMemcpy(M, d_M, sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles, cudaMemcpyDeviceToHost);
    // 输出转换
    output_transform(M, Y, ti, us.oc * vs.num_tiles);
    output_unpacking_store(Y, out, os, ti);

    // 释放 CPU 内存
    free(packed_filter);
    free(packed_image);
    free(U);
    free(V);
    free(M);
    free(Y);

    // 释放 GPU 内存
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_M);

    // 销毁 cuBLAS 句柄
    cublasDestroy(handle);
}