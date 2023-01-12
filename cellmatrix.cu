__global__ void
  copyCUDAKernel(int     nVec,
                 int     numCellsTimesnumPointsPerCell,
                 double *x,
                 double *U,
                 int *   map)
  {
    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
    int numberEntries  = nVec * numCellsTimesnumPointsPerCell;
    if (globalThreadId < numberEntries)
      {
        int blockIndex      = globalThreadId / nVec;
        int intraBlockIndex = globalThreadId % nVec;
        U[globalThreadId]   = x[nVec * map[blockIndex] + intraBlockIndex];
      }
  }

  __global__ void
  assemblyAtomic(int     nVec,
                 int     numCellsTimesnumPointsPerCell,
                 double *V,
                 double *Ax,
                 int *   map)
  {
    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
    int numberEntries  = nVec * numCellsTimesnumPointsPerCell;
    if (globalThreadId < numberEntries)
      {
        int blockIndex      = globalThreadId / nVec;
        int intraBlockIndex = globalThreadId % nVec;
        atomicAdd(&Ax[nVec * map[blockIndex] + intraBlockIndex],
                  V[globalThreadId]);
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::computeAXCellMatrix(
    distributedGPUVec<double> &Ax,
    distributedGPUVec<double> &x,
    const int &                nVec)
  {
    constexpr int    BLOCK_SIZE    = 512;
    constexpr int    p             = FEOrderElectro + 1;
    constexpr int    dofs_per_cell = p * p * p;
    constexpr double alpha         = 1.0;
    constexpr double beta          = 0.0;

    constexpr int vecShared = 4;

    constexpr int yThreads = (FEOrderElectro < 8 ? 64 : 128);
    const int     batch    = (nVec == 1) ? 1 : nVec / vecShared;

    dim3 blocks(d_nLocalCells, batch, 1);
    dim3 threads(vecShared, yThreads, 1);

    constexpr size_t smem =
      4 * vecShared * p * p * p * sizeof(double) + 4 * p * p * sizeof(double) +
      3 * 3 * sizeof(double); // + p * p * p * sizeof(int);


    cudaMemset(Ax.begin(), 0, nVec * d_xLen * sizeof(double));

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistribute(x);

    x.updateGhostValues();

    d_constraintsTotalPotentialInfoCellMatrix.distribute(x, nVec);

    copyCUDAKernel<<<(nVec * d_nLocalCells * dofs_per_cell) / BLOCK_SIZE + 1,
                     BLOCK_SIZE>>>(
      nVec, d_nLocalCells * dofs_per_cell, x.begin(), U_ptr, mapCellmatrixPtr);

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec,
                                          dofs_per_cell,
                                          dofs_per_cell,
                                          &alpha,
                                          U_ptr,
                                          nVec,
                                          nVec * dofs_per_cell,
                                          cellMatrixPtr,
                                          dofs_per_cell,
                                          dofs_per_cell * dofs_per_cell,
                                          &beta,
                                          V_ptr,
                                          nVec,
                                          nVec * dofs_per_cell,
                                          d_nLocalCells)); //*/

    // Old MatrixFree Shared Memory Implementation
    /*sharedFusedKernel<p, 4>
      <<<blocks, threads, smem>>>(shapeFunctionValuePtr,
                                  shapeFunctionGradientPtr,
                                  x.begin(),
                                  Ax.begin(),
                                  mapCellmatrixPtr,
                                  detJ_ptr,
                                  nVec,
                                  d_nLocalCells,
                                  d_nLocalCells * nVec / (vecShared)); //*/

    // MatrixFree with OldLayout
    /*computeAXKernel<double, p * p, p, p, 3>
      <<<blocks, threads, smem>>>(Ax.begin(),
                                  x.begin(),
                                  shapeFunctionAllPtr,
                                  jacobianActionPtr,
                                  mapCellmatrixPtr,
                                  d_coeffHelmholtz,
                                  vecShared); //*/

    // MatrixFree cuBLAS
    /*cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionValuePtr,
                                          p,
                                          0,
                                          &beta,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionValuePtr,
                                          p,
                                          0,
                                          &beta,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionValuePtr,
                                          p,
                                          0,
                                          &beta,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionGradientPtr,
                                          p,
                                          0,
                                          &beta,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionGradientPtr,
                                          p,
                                          0,
                                          &beta,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionGradientPtr,
                                          p,
                                          0,
                                          &beta,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionGradientPtr,
                                          p,
                                          0,
                                          &beta,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionGradientPtr,
                                          p,
                                          0,
                                          &beta,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionGradientPtr,
                                          p,
                                          0,
                                          &beta,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionValuePtr,
                                          p,
                                          0,
                                          &beta,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionValuePtr,
                                          p,
                                          0,
                                          &beta,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells));

    cublasCheck(cublasDgemmStridedBatched(*d_cublasHandlePtr,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          nVec * p * p,
                                          p,
                                          p,
                                          &alpha,
                                          V_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          shapeFunctionValuePtr,
                                          p,
                                          0,
                                          &beta,
                                          U_ptr,
                                          nVec * p * p,
                                          nVec * p * p * p,
                                          d_nLocalCells)); //*/


    assemblyAtomic<<<(nVec * d_nLocalCells * dofs_per_cell) / BLOCK_SIZE + 1,
                     BLOCK_SIZE>>>(
      nVec, d_nLocalCells * dofs_per_cell, U_ptr, Ax.begin(), mapCellmatrixPtr);

    d_constraintsTotalPotentialInfoCellMatrix.distribute_slave_to_master(Ax,
                                                                         nVec);

    Ax.compressAdd();

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistributeSlaveToMaster(Ax);
  }