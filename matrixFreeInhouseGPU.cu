#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <time.h>

#include <cmath>
#include <fstream>
#include <iostream>

extern "C" {
#include "cuda_profiler_api.h"
}

// Checks for Memory Allocation on Device
#define cudaCheck(expr)                                                                                                                                                                                              \
	{                                                                                                                                                                                                                \
		cudaError_t __cuda_error = expr;                                                                                                                                                                             \
		if ((__cuda_error) != cudaSuccess) {                                                                                                                                                                         \
			std::cout << "CUDA error on or before line number " << __LINE__ << " in file: " << __FILE__ << ". Error code: " << (__cuda_error) << ". Description: " << cudaGetErrorString(__cuda_error) << std::endl; \
			std::cout << "Terminating execution..." << std::endl;                                                                                                                                                    \
			cudaDeviceReset();                                                                                                                                                                                       \
			exit(0);                                                                                                                                                                                                 \
		}                                                                                                                                                                                                            \
	}

template <typename T>
struct square {
	__host__ __device__
		T
		operator()(const T &x) const {
		return x * x;
	}
};

template <typename Type, int M, int N, int K, int dim>
__global__ void
computeAXKernel(Type *V,
				const Type *U,
				const Type *P,
				const Type *J,
				const int *map,
				Type coeffHelmholtz,
				int vecShared) {
	// V = AU
	// gridDim.x = cells;
	// gridDim.y = batch;
	// nVec = vecShared * batch;
	// vecShared -> No of vectors in shared memory
	// First index is fastest convention used
	// sharedT is used to temporarily store UP^T/UP
	// PT(q*p), D(q*q), P(p*q), DT(q*q)

	extern __shared__ Type SMem[];

	Type *sharedX = SMem;
	Type *sharedY = &sharedX[vecShared * N * N * N];
	Type *sharedZ = &sharedY[vecShared * N * N * N];
	Type *sharedT = &sharedZ[vecShared * N * N * N];
	Type *sharedPT = &sharedT[vecShared * N * N * N];
	Type *sharedD = &sharedPT[N * K];
	Type *sharedP = &sharedD[N * N];
	Type *sharedDT = &sharedP[K * N];
	Type *sharedJ = &sharedDT[N * N];
	// int * sharedMap = (int *)&sharedJ[dim * dim];

	const int mapShift = (blockIdx.x + blockIdx.y * gridDim.x) * M * K;

	// Copy Map to shared memory
	// #pragma unroll
	//     for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < M * K;
	//          i += blockDim.x * blockDim.y)
	//       sharedMap[i] = map[i + mapShift];

	// Copy Shape Function Values and Gradients to shared memory
#pragma unroll
	for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 2 * N * (K + N);
		 i += blockDim.x * blockDim.y)
		sharedPT[i] = P[i];

	__syncthreads();

	//////////////////////////////////////////////////////////////
	// First index is the fastest
	// Interpolation combined with Extraction
	// T -> PPPU
	// X -> TD1
	// Y -> TD2
	// Z -> TD3

	// 1st GEMM of P
	// Z Direction
	for (int i = threadIdx.y; i < M; i += blockDim.y) {
		Type x[N], u[K];

#pragma unroll
		for (int j = 0; j < N; j++)
			x[j] = 0.0;

		for (int k = 0; k < K; k++) {
			u[k] = U[threadIdx.x + map[i + k * M + mapShift]];

#pragma unroll
			for (int j = 0; j < N; j++)
				x[j] += sharedPT[j + k * N] * u[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedX[threadIdx.x + i * vecShared + j * vecShared * M] = x[j];
	}

	__syncthreads();

	// 2nd GEMM of P
	// Y Direction
	for (int i = threadIdx.y; i < K * N; i += blockDim.y) {
		Type y[N], x[K];

		int a = i % K;
		int b = i / K;

#pragma unroll
		for (int j = 0; j < N; j++)
			y[j] = 0.0;

		for (int k = 0; k < K; k++) {
			x[k] = sharedX[threadIdx.x + a * vecShared + k * vecShared * K +
						   b * vecShared * M];

#pragma unroll
			for (int j = 0; j < N; j++)
				y[j] += sharedPT[j + k * N] * x[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedY[threadIdx.x + a * vecShared + j * vecShared * K +
					b * vecShared * K * N] = y[j];
	}

	__syncthreads();

	// 3rd GEMM of P
	// X Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type x[N], y[K];

#pragma unroll
		for (int j = 0; j < N; j++)
			x[j] = 0.0;

		for (int k = 0; k < K; k++) {
			y[k] = sharedY[threadIdx.x + k * vecShared + i * vecShared * K];

#pragma unroll
			for (int j = 0; j < N; j++)
				x[j] += sharedPT[j + k * N] * y[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedX[threadIdx.x + j * vecShared + i * vecShared * N] = x[j];
	}

	__syncthreads();

	// 1st GEMM of D
	// Z Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type y[N], x[N];

#pragma unroll
		for (int j = 0; j < N; j++)
			y[j] = 0.0;

		for (int k = 0; k < N; k++) {
			x[k] =
				sharedX[threadIdx.x + i * vecShared + k * vecShared * N * N];

#pragma unroll
			for (int j = 0; j < N; j++)
				y[j] += sharedD[j + k * N] * x[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedY[threadIdx.x + i * vecShared + j * vecShared * N * N] = y[j];
	}

	// 2nd GEMM of D
	// Y Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type z[N], x[N];

		int a = i % N;
		int b = i / N;

#pragma unroll
		for (int j = 0; j < N; j++)
			z[j] = 0.0;

		for (int k = 0; k < N; k++) {
			x[k] = sharedX[threadIdx.x + a * vecShared + k * vecShared * N +
						   b * vecShared * N * N];

#pragma unroll
			for (int j = 0; j < N; j++)
				z[j] += sharedD[j + k * N] * x[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedZ[threadIdx.x + a * vecShared + j * vecShared * N +
					b * vecShared * N * N] = z[j];
	}

	// 3rd GEMM of D
	// X Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type t[N], x[N];

#pragma unroll
		for (int j = 0; j < N; j++)
			t[j] = 0.0;

		for (int k = 0; k < N; k++) {
			x[k] = sharedX[threadIdx.x + k * vecShared + i * vecShared * N];

#pragma unroll
			for (int j = 0; j < N; j++)
				t[j] += sharedD[j + k * N] * x[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedT[threadIdx.x + j * vecShared + i * vecShared * N] = t[j];
	}

	//////////////////////////////////////////////////////////////////
	// sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
	// directions

	// Copy Jacobian Action to shared memory
#pragma unroll
	for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < dim * dim;
		 i += blockDim.x * blockDim.y)
		sharedJ[i] = J[i + blockIdx.x * dim * dim];

	Type detJ;

	__syncthreads();

	// Gemm with Jacobian Action
#pragma unroll
	for (int i = threadIdx.y; i < N * N * N; i += blockDim.y) {
		Type v[3];

		v[2] = sharedY[threadIdx.x + i * vecShared];
		v[1] = sharedZ[threadIdx.x + i * vecShared];
		v[0] = sharedT[threadIdx.x + i * vecShared];

		sharedY[threadIdx.x + i * vecShared] =
			sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
		sharedZ[threadIdx.x + i * vecShared] =
			sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
		sharedT[threadIdx.x + i * vecShared] =
			sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];

		detJ =
			sharedJ[0] * (sharedJ[4] * sharedJ[8] - sharedJ[5] * sharedJ[7]) -
			sharedJ[1] * (sharedJ[3] * sharedJ[8] - sharedJ[5] * sharedJ[6]) +
			sharedJ[2] * (sharedJ[3] * sharedJ[7] - sharedJ[4] * sharedJ[6]);
	}

	__syncthreads();

	// Integration
	// X -> TDT1
	// Y -> TDT2
	// Z -> TDT3
	// T -> PPPU

	// 1st GEMM of DT
	// Z Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type x[N], y[N], h[N];

#pragma unroll
		for (int j = 0; j < N; j++)
			x[j] = 0.0;

		for (int k = 0; k < N; k++) {
			y[k] =
				sharedY[threadIdx.x + i * vecShared + k * vecShared * N * N];

#pragma unroll
			for (int j = 0; j < N; j++)
				x[j] += sharedDT[j + k * N] * y[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++) {
			h[j] =
				sharedX[threadIdx.x + i * vecShared + j * vecShared * N * N];
			sharedX[threadIdx.x + i * vecShared + j * vecShared * N * N] =
				coeffHelmholtz * detJ * h[j] + x[j];
		}
	}

	__syncthreads();

	// 2nd GEMM of DT
	// Y Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type y[N], z[N];

		int a = i % N;
		int b = i / N;

#pragma unroll
		for (int j = 0; j < N; j++)
			y[j] = 0.0;

		for (int k = 0; k < N; k++) {
			z[k] = sharedZ[threadIdx.x + a * vecShared + k * vecShared * N +
						   b * vecShared * N * N];

#pragma unroll
			for (int j = 0; j < N; j++)
				y[j] += sharedDT[j + k * N] * z[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedX[threadIdx.x + a * vecShared + j * vecShared * N +
					b * vecShared * N * N] += y[j];
	}

	__syncthreads();

	// 3rd GEMM of DT
	// X Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type z[N], t[N];

#pragma unroll
		for (int j = 0; j < N; j++)
			z[j] = 0.0;

		for (int k = 0; k < N; k++) {
			t[k] = sharedT[threadIdx.x + k * vecShared + i * vecShared * N];

#pragma unroll
			for (int j = 0; j < N; j++)
				z[j] += sharedDT[j + k * N] * t[k];
		}

#pragma unroll
		for (int j = 0; j < N; j++)
			sharedX[threadIdx.x + j * vecShared + i * vecShared * N] += z[j];
	}

	__syncthreads();

	// 1st GEMM of PT
	// Z Direction
	for (int i = threadIdx.y; i < N * N; i += blockDim.y) {
		Type y[K], x[N];

#pragma unroll
		for (int j = 0; j < K; j++)
			y[j] = 0.0;

		for (int k = 0; k < N; k++) {
			x[k] =
				sharedX[threadIdx.x + i * vecShared + k * vecShared * N * N];

#pragma unroll
			for (int j = 0; j < K; j++)
				y[j] += sharedP[j + k * K] * x[k];
		}

#pragma unroll
		for (int j = 0; j < K; j++)
			sharedY[threadIdx.x + i * vecShared + j * vecShared * N * N] = y[j];
	}

	__syncthreads();

	// 2nd GEMM of PT
	// Y Direction
	for (int i = threadIdx.y; i < N * K; i += blockDim.y) {
		Type x[K], y[N];

		int a = i % N;
		int b = i / N;

#pragma unroll
		for (int j = 0; j < K; j++)
			x[j] = 0.0;

		for (int k = 0; k < N; k++) {
			y[k] = sharedY[threadIdx.x + a * vecShared + k * vecShared * N +
						   b * vecShared * N * N];

#pragma unroll
			for (int j = 0; j < K; j++)
				x[j] += sharedP[j + k * K] * y[k];
		}

#pragma unroll
		for (int j = 0; j < K; j++)
			sharedX[threadIdx.x + a * vecShared + j * vecShared * N +
					b * vecShared * N * K] = x[j];
	}

	__syncthreads();

	// 3rd GEMM of PT
	// X Direction
	for (int i = threadIdx.y; i < M; i += blockDim.y) {
		Type y[K], x[N];

#pragma unroll
		for (int j = 0; j < K; j++)
			y[j] = 0.0;

		for (int k = 0; k < N; k++) {
			x[k] = sharedX[threadIdx.x + k * vecShared + i * vecShared * N];

#pragma unroll
			for (int j = 0; j < K; j++)
				y[j] += sharedP[j + k * K] * x[k];
		}

#pragma unroll
		for (int j = 0; j < K; j++)
			atomicAdd(&V[threadIdx.x + map[j + i * K + mapShift]], y[j]);
	}
}

int main() {
	constexpr int p = 9;
	constexpr int q = p;
	constexpr int nVec = 1024;
	constexpr int vecShared = 4;
	constexpr int dim = 3;
	constexpr int cells = 216;
	constexpr int Dofs = 117649;
	constexpr double gamma = 0.5;
	constexpr double coeffHelmholtz = 4 * M_PI * gamma;

	constexpr int yThreads = (p < 9 ? 64 : 128);
	constexpr int batch = (nVec == 1) ? 1 : nVec / vecShared;

	std::cout << "p         : " << p << "\n"
			  << "vecShared : " << vecShared << "\n"
			  << "nVec      : " << nVec << "\n"
			  << "Dofs      : " << Dofs << "\n"
			  << "cells     : " << cells << "\n";

	// Initialize the Seed
	srand(unsigned(std::time(0)));

	thrust::host_vector<double> P(2 * p * q + 2 * q * q), J(dim * dim * cells), x(nVec * Dofs);
	thrust::host_vector<int> map(p * p * p * cells * batch);

	thrust::device_vector<double> dev_P, dev_J, dev_x, dev_Ax(nVec * Dofs);
	thrust::device_vector<int> dev_map;

	for (int i = 0; i < P.size(); i++)
		P[i] = (double)rand() / RAND_MAX;

	for (int i = 0; i < J.size(); i++)
		J[i] = (double)rand() / RAND_MAX;

	for (int i = 0; i < x.size(); i++)
		x[i] = (double)rand() / RAND_MAX;

	std::ifstream inFile;
	inFile.open("map.txt", std::ios::in);
	int element;
	int i = 0;
	while (inFile >> element) {
		map[i] = element;
		i++;
	}
	inFile.close();

	dev_x = x;
	dev_P = P;
	dev_J = J;
	dev_map = map;

	double *dev_Ax_ptr = thrust::raw_pointer_cast(dev_Ax.data());
	double *dev_x_ptr = thrust::raw_pointer_cast(dev_x.data());
	double *dev_P_ptr = thrust::raw_pointer_cast(dev_P.data());
	double *dev_J_ptr = thrust::raw_pointer_cast(dev_J.data());
	int *dev_map_ptr = thrust::raw_pointer_cast(dev_map.data());

	const size_t smem =
		4 * vecShared * p * p * p * sizeof(double) +
		4 * p * p * sizeof(double) +
		dim * dim * sizeof(double);	 // + p * p * p * sizeof(int);

	cudaFuncSetAttribute(computeAXKernel<double, p * p, q, p, dim>,
						 cudaFuncAttributeMaxDynamicSharedMemorySize,
						 smem);

	// Kernel parameters
	// Blocks and threads initialization
	dim3 blocks(cells, batch, 1);
	dim3 threads(vecShared, yThreads, 1);

	cudaProfilerStart();

	computeAXKernel<double, p * p, q, p, dim>
		<<<blocks, threads, smem>>>(dev_Ax_ptr,
									dev_x_ptr,
									dev_P_ptr,
									dev_J_ptr,
									dev_map_ptr,
									coeffHelmholtz,
									vecShared);

	cudaProfilerStop();

	cudaCheck(cudaPeekAtLastError());

	square<double>        unary_op;
    thrust::plus<double> binary_op;
    double init = 0.;

    // compute norm
    double norm = std::sqrt( thrust::transform_reduce(dev_Ax.begin(), dev_Ax.end(), unary_op, init, binary_op) );

    std::cout << "L2 Norm of Ax: " << norm << std::endl;

	return 0;
}
