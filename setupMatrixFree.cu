template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::setupMatrixFree(
    const int & vecShared,
    const int & nVec,
    const int & nQuad,
    const bool &MFflag,
    const bool &cellFlag,
    const bool &accuracyFlag)
  {
    constexpr int    p              = FEOrderElectro + 1;
    constexpr int    dim            = 3;
    constexpr int    q              = p;
    constexpr double gamma          = 0.5;
    constexpr double coeffLaplacian = 1.0 / (4.0 * M_PI);
    d_coeffHelmholtz                = 4 * M_PI * gamma;

    // const int vecShared = (nVec == 1) ? 1 : (FEOrderElectro == 7 ? 5 : 4);

    // shape info helps to obtain reference cell basis function and lex
    // numbering
    const dealii::DoFHandler<dim> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);
    const int dofs_per_cell = dofHandler.get_fe().dofs_per_cell;

    dealii::QGauss<dim> quadrature_formula(dofHandler.get_fe().degree + 1);
    const int           qPoints = quadrature_formula.size();

    dealii::FEValues<dim> fe_values(dofHandler.get_fe(),
                                    quadrature_formula,
                                    dealii::update_inverse_jacobians |
                                      dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_JxW_values |
                                      dealii::update_quadrature_points);

    if (MFflag)
      {
        const int batch = nVec / vecShared;

        dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shapeInfo;

        // const dealii::Quadrature<dim> &quadrature =
        // QIterated<dim>(QGauss<1>(q));

        const dealii::Quadrature<dim> &quadrature =
          d_matrixFreeDataPtr->get_quadrature(
            d_matrixFreeQuadratureComponentAX);

        int num_quapoints = std::cbrt(quadrature.size());

        dealii::QGauss<1> quad(num_quapoints);
        shapeInfo.reinit(quad, dofHandler.get_fe());
        std::vector<unsigned int> lexMap3D = shapeInfo.lexicographic_numbering;

        const auto shapeValue = shapeInfo.data.front().shape_values;
        const auto shapeGrad  = shapeInfo.data.front().shape_gradients;
        const auto shapeGradquad =
          shapeInfo.data.front().shape_gradients_collocation;

        dealii::FE_Q<1> feCell1D(FEOrderElectro);
        shapeInfo.reinit(quad, feCell1D);
        std::vector<unsigned int> lexMap1D = shapeInfo.lexicographic_numbering;
        std::vector<double>       quadWeights(p);

        for (int i = 0; i < p; i++)
          quadWeights[i] = quad.weight(lexMap1D[i]);

        thrust::host_vector<double> spV(p * q), spG(p * q),
          spVG(2 * p * q + 2 * q * q);

        for (int i = 0; i < p; i++)
          for (int j = 0; j < q; j++)
            {
              spV[i + j * p] =
                shapeValue[lexMap1D[j] + i * p] * std::sqrt(quadWeights[j]);
              spG[i + j * p] =
                shapeGrad[lexMap1D[j] + i * p] * std::sqrt(quadWeights[j]);
            }

        for (int i = 0; i < p; i++)
          for (int j = 0; j < q; j++)
            {
              // PT(q*p), DT(q*q), P(p*q), D(q*q)
              double value =
                shapeValue[lexMap1D[j] + i * p] * std::sqrt(quadWeights[j]);
              double grad = shapeGradquad[lexMap1D[j] + lexMap1D[i] * p] *
                            std::sqrt(quadWeights[j]) /
                            std::sqrt(quadWeights[i]);

              spVG[j + i * q]                     = value;
              spVG[j + i * q + q * p]             = grad;
              spVG[i + j * p + q * p + q * q]     = value;
              spVG[i + j * q + 2 * q * p + q * q] = grad;
            }

        dealii::Triangulation<1> reference_cell;
        dealii::GridGenerator::hyper_cube(reference_cell, 0, 1);
        dealii::FEValues<1> fe_values_reference(feCell1D,
                                                quad,
                                                dealii::update_values |
                                                  dealii::update_gradients |
                                                  dealii::update_JxW_values);

        fe_values_reference.reinit(reference_cell.begin());

        // Map making
        thrust::host_vector<int> map(dofs_per_cell * d_nLocalCells);
        std::vector<dealii::types::global_dof_index> local_dof_globalIndices(
          dofs_per_cell);

        // Lexicographic Map making
        int cellIdx = 0;
        for (const auto &cell : dofHandler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(local_dof_globalIndices);

                for (int dofIdx = 0; dofIdx < dofs_per_cell; dofIdx++)
                  {
                    dealii::types::global_dof_index globalIdx =
                      local_dof_globalIndices[lexMap3D[dofIdx]];
                    int localIdx =
                      d_xPtr->get_partitioner()->global_to_local(globalIdx);
                    map[dofIdx + cellIdx * dofs_per_cell] = localIdx;
                  }
                cellIdx++;
              }
          }

        thrust::host_vector<int> map_newlayout(dofs_per_cell * d_nLocalCells *
                                               batch);
        auto         taskGhostMap = d_xPtr->get_partitioner()->ghost_targets();
        unsigned int n_procs =
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
        std::vector<unsigned int> taskGhostStartIndices(n_procs, 0);

        for (unsigned int i = 0; i < taskGhostMap.size(); ++i)
          {
            taskGhostStartIndices[taskGhostMap[i].first] =
              taskGhostMap[i].second;
          }

        unsigned int ghostSum = 0;

        for (unsigned int i = 0; i < taskGhostStartIndices.size(); ++i)
          {
            unsigned int tmp = ghostSum;
            ghostSum += taskGhostStartIndices[i];
            taskGhostStartIndices[i] = tmp;
          }

        for (unsigned int iCell = 0; iCell < d_nLocalCells; ++iCell)
          {
            for (unsigned int ildof = 0; ildof < dofs_per_cell; ++ildof)
              {
                unsigned int l2g = map[ildof + dofs_per_cell * iCell];
                if (l2g >= d_xLocalDof)
                  {
                    unsigned int ownerId = 0;
                    while (taskGhostStartIndices[ownerId] <= l2g - d_xLocalDof)
                      {
                        ++ownerId;
                        if (ownerId == n_procs)
                          break;
                      }

                    --ownerId;
                    unsigned int ghostIdFromOwner =
                      l2g - taskGhostStartIndices[ownerId] - d_xLocalDof;
                    unsigned int nGhostsFromOwner =
                      ownerId == n_procs - 1 ?
                        d_xPtr->get_partitioner()->n_ghost_indices() -
                          taskGhostStartIndices[ownerId] :
                        taskGhostStartIndices[ownerId + 1] -
                          taskGhostStartIndices[ownerId];

                    for (unsigned int ibatch = 0; ibatch < batch; ++ibatch)
                      {
                        map_newlayout[ildof + dofs_per_cell * iCell +
                                      ibatch * dofs_per_cell * d_nLocalCells] =
                          (d_xLocalDof + taskGhostStartIndices[ownerId]) *
                            nVec +
                          ghostIdFromOwner * vecShared +
                          ibatch * nGhostsFromOwner * vecShared;
                      }
                  }
                else
                  {
                    for (unsigned int ibatch = 0; ibatch < batch; ++ibatch)
                      map_newlayout[ildof + dofs_per_cell * iCell +
                                    ibatch * dofs_per_cell * d_nLocalCells] =
                        l2g * vecShared + ibatch * d_xLocalDof * vecShared;
                  }
              }
          }

        d_mapNew  = map_newlayout;
        mapNewPtr = thrust::raw_pointer_cast(d_mapNew.data());

        /*if (this_mpi_process == 0)
          {
            std::ofstream mapFile("map8.txt");
            if (mapFile.is_open())
              {
                for (unsigned int i = 0; i < map_newlayout.size(); ++i)
                  mapFile << std::setprecision(
                               std::numeric_limits<double>::max_digits10)
                          << map_newlayout[i] << "\n";

                mapFile.close();
              }
          } //*/

        std::vector<dealii::DerivativeForm<1, dim, dim>> inv_jacobians_tensor;
        std::vector<double> detJacobian(d_nLocalCells * qPoints),
          invJac(d_nLocalCells * dim * dim);
        thrust::host_vector<double> jacobianAction(d_nLocalCells * dim * dim),
          detJ(d_nLocalCells);

        cellIdx = 0;
        for (const auto &cell : dofHandler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                inv_jacobians_tensor = fe_values.get_inverse_jacobians();

                for (int d = 0; d < dim; d++)
                  for (int e = 0; e < dim; e++)
                    invJac[e + d * dim + cellIdx * dim * dim] =
                      inv_jacobians_tensor[0][d][e];

                for (int i = 0; i < qPoints; i++)
                  detJacobian[i + cellIdx * qPoints] =
                    fe_values.JxW(lexMap3D[i]) /
                    quadrature_formula.weight(
                      lexMap3D[i]); // MassMatrix and Helmholtz
                                    // quadrature_formula.weight(lexMap3D[i]) *
                                    // coeffLaplacian; // Laplacian
                cellIdx++;
              }
          }

        for (int cellIdx = 0; cellIdx < d_nLocalCells; cellIdx++)
          detJ[cellIdx] = detJacobian[cellIdx * dofs_per_cell];

        for (int cellIdx = 0; cellIdx < d_nLocalCells; cellIdx++)
          for (int d = 0; d < dim; d++)
            for (int e = 0; e < dim; e++)
              for (int f = 0; f < dim; f++)
                jacobianAction[e + d * dim + cellIdx * dim * dim] +=
                  invJac[f + d * dim + cellIdx * dim * dim] *
                  invJac[e + f * dim + cellIdx * dim * dim] *
                  detJacobian[cellIdx * qPoints];

        // Construct the device vectors
        d_shapeFunctionValue    = spV;
        d_shapeFunctionGradient = spG;
        d_shapeFunctionAll      = spVG;
        d_jacobianAction        = jacobianAction;
        d_map                   = map;
        d_detJ                  = detJ;

        shapeFunctionValuePtr =
          thrust::raw_pointer_cast(d_shapeFunctionValue.data());
        shapeFunctionGradientPtr =
          thrust::raw_pointer_cast(d_shapeFunctionGradient.data());
        shapeFunctionAllPtr =
          thrust::raw_pointer_cast(d_shapeFunctionAll.data());
        jacobianActionPtr = thrust::raw_pointer_cast(d_jacobianAction.data());
        detJ_ptr          = thrust::raw_pointer_cast(d_detJ.data());
        mapPtr            = thrust::raw_pointer_cast(d_map.data());

        const size_t smem =
          4 * vecShared * p * p * p * sizeof(double) +
          4 * p * p * sizeof(double) +
          dim * dim * sizeof(double); // + p * p * p * sizeof(int);

        cudaFuncSetAttribute(computeAXKernel<double, p * p, q, p, dim>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem);
      }

    if (cellFlag)
      {
        thrust::host_vector<int> map_cellmatrix(dofs_per_cell * d_nLocalCells);
        std::vector<dealii::types::global_dof_index> local_dof_globalIndices(
          dofs_per_cell);

        int cellIdx = 0;
        for (const auto &cell : dofHandler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(local_dof_globalIndices);

                for (int dofIdx = 0; dofIdx < dofs_per_cell; dofIdx++)
                  {
                    dealii::types::global_dof_index globalIdx =
                      local_dof_globalIndices[dofIdx];
                    int localIdx =
                      d_xPtr->get_partitioner()->global_to_local(globalIdx);
                    map_cellmatrix[dofIdx + cellIdx * dofs_per_cell] = localIdx;
                  }
                cellIdx++;
              }
          }

        dealii::FullMatrix<double>  cell_matrix(dofs_per_cell, dofs_per_cell);
        thrust::host_vector<double> H(dofs_per_cell * dofs_per_cell *
                                      d_nLocalCells);

        if (accuracyFlag)
          {
            cellIdx = 0;
            for (const auto &cell : dofHandler.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  {
                    fe_values.reinit(cell);

                    cell_matrix = 0;
                    for (int i = 0; i < dofs_per_cell; i++)
                      for (int j = 0; j < dofs_per_cell; j++)
                        for (int k = 0; k < qPoints; k++)
                          {
                            // shape_value for MassMatrix and shape_grad for
                            // Laplacian and Helmholtz

                            // Helmholtz
                            cell_matrix(i, j) +=
                              fe_values.shape_grad(i, k) *
                                fe_values.shape_grad(j, k) * fe_values.JxW(k) +
                              d_coeffHelmholtz * fe_values.shape_value(i, k) *
                                fe_values.shape_value(j, k) * fe_values.JxW(k);

                            // Laplacian
                            // cell_matrix(i, j) += fe_values.shape_grad(i, k) *
                            //  fe_values.shape_grad(j, k) *
                            //  fe_values.JxW(k) * coeffLaplacian;

                            // MassMatrix
                            // cell_matrix(i, j) += fe_values.shape_value(i, k)
                            // *
                            //                     fe_values.shape_value(j, k) *
                            //                     fe_values.JxW(k);
                          }

                    for (int i = 0; i < dofs_per_cell; i++)
                      for (int j = 0; j < dofs_per_cell; j++)
                        {
                          H[j + i * dofs_per_cell +
                            cellIdx * dofs_per_cell * dofs_per_cell] =
                            cell_matrix(i, j);
                        }

                    cellIdx++;
                  }
              }
          }
        else
          {
            for (int idx = 0; idx < d_nLocalCells; idx++)
              for (int i = 0; i < dofs_per_cell; i++)
                for (int j = 0; j < dofs_per_cell; j++)
                  {
                    H[j + i * dofs_per_cell +
                      idx * dofs_per_cell * dofs_per_cell] =
                      (double)rand() / RAND_MAX;
                  }
          }

        // thrust::host_vector<double> spV(p * q), spG(p * q),
        //   detJtemp(3 * 3 * d_nLocalCells);

        // for (int i = 0; i < p; i++)
        //   for (int j = 0; j < q; j++)
        //     {
        //       double val     = (double)rand() / RAND_MAX;
        //       spV[i + j * p] = val;
        //       spG[i + j * p] = val;
        //     }

        // for (int i = 0; i < 3 * 3 * d_nLocalCells; i++)
        //   detJtemp[i] = (double)rand() / RAND_MAX;

        // d_shapeFunctionValue    = spV;
        // d_shapeFunctionGradient = spG;
        // d_detJ                  = detJtemp;

        // shapeFunctionValuePtr =
        //   thrust::raw_pointer_cast(d_shapeFunctionValue.data());
        // shapeFunctionGradientPtr =
        //   thrust::raw_pointer_cast(d_shapeFunctionGradient.data());
        // detJ_ptr = thrust::raw_pointer_cast(d_detJ.data());

        // constexpr size_t smem =
        //   4 * 4 * p * p * p * sizeof(double) + 2 * p * p * sizeof(double) +
        //   dim * dim * sizeof(double); // + p * p * p * sizeof(int);

        // cudaFuncSetAttribute(sharedFusedKernel<p, 4>,
        //                      cudaFuncAttributeMaxDynamicSharedMemorySize,
        //                      smem);

        d_U.resize(nVec * dofs_per_cell * d_nLocalCells);
        d_V.resize(nVec * dofs_per_cell * d_nLocalCells);

        d_cellMatrix     = H;
        d_map_cellmatrix = map_cellmatrix;

        cellMatrixPtr    = thrust::raw_pointer_cast(d_cellMatrix.data());
        U_ptr            = thrust::raw_pointer_cast(d_U.data());
        V_ptr            = thrust::raw_pointer_cast(d_V.data());
        mapCellmatrixPtr = thrust::raw_pointer_cast(d_map_cellmatrix.data());
      }
  }