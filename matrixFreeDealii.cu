template <unsigned int FEOrder, unsigned int FEOrderElectro>
void poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::computeAXDealii(
	distributedGPUVec<double> &Ax,
	distributedGPUVec<double> &x) {
	using dealiiVector =
		dealii::LinearAlgebra::distributed::Vector<double,
												   dealii::MemorySpace::CUDA>;

	dealiiVector *AxDealii = (dealiiVector *)Ax.getDealiiVec();
	dealiiVector *xDealii = (dealiiVector *)x.getDealiiVec();

	if (d_isMeanValueConstraintComputed)
		meanValueConstraintDistribute(x);

	x.updateGhostValues();

	vmult(*AxDealii, *xDealii);

	Ax.compressAdd();

	if (d_isMeanValueConstraintComputed)
		meanValueConstraintDistributeSlaveToMaster(Ax);
}

template <int dim, int fe_degree>
class VaryingCoefficientFunctor {
   public:
	VaryingCoefficientFunctor(double *coefficient)
		: coef(coefficient) {}

	__device__ void
	operator()(
		const unsigned int cell,
		const typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data
			*gpu_data);

	static const unsigned int n_dofs_1d = fe_degree + 1;
	static const unsigned int n_local_dofs =
		dealii::Utilities::pow(n_dofs_1d, dim);
	static const unsigned int n_q_points =
		dealii::Utilities::pow(n_dofs_1d, dim);

   private:
	double *coef;
};

template <int dim, int fe_degree>
__device__ void
VaryingCoefficientFunctor<dim, fe_degree>::
operator()(const unsigned int cell,
		   const typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data
			   *gpu_data) {
	constexpr double gamma = 0.5;
	constexpr double coeffHelmholtz = 4 * M_PI * gamma;

	const unsigned int pos =
		dealii::CUDAWrappers::local_q_point_id<dim, double>(cell,
															gpu_data,
															n_dofs_1d,
															n_q_points);

	coef[pos] = coeffHelmholtz;
}

template <int dim, int fe_degree>
class HelmholtzOperatorQuad {
   public:
	__device__
	HelmholtzOperatorQuad(double coef)
		: coef(coef) {}

	__device__ void operator()(
		dealii::CUDAWrappers::
			FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> *fe_eval) const;

   private:
	double coef;
};

template <int dim, int fe_degree>
__device__ void HelmholtzOperatorQuad<dim, fe_degree>::operator()(
	dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
		*fe_eval) const {
	fe_eval->submit_value(coef * fe_eval->get_value());
	fe_eval->submit_gradient(fe_eval->get_gradient());
}

template <int dim, int fe_degree>
class LocalHelmholtzOperator {
   public:
	LocalHelmholtzOperator(double *coefficient)
		: coef(coefficient) {}

	__device__ void
	operator()(
		const unsigned int cell,
		const typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data
			*gpu_data,
		dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
		const double *src,
		double *dst) const;

	static const unsigned int n_dofs_1d = fe_degree + 1;
	static const unsigned int n_local_dofs =
		dealii::Utilities::pow(fe_degree + 1, dim);
	static const unsigned int n_q_points =
		dealii::Utilities::pow(fe_degree + 1, dim);

   private:
	double *coef;
};

template <int dim, int fe_degree>
__device__ void
LocalHelmholtzOperator<dim, fe_degree>::
operator()(const unsigned int cell,
		   const typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data
			   *gpu_data,
		   dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
		   const double *src,
		   double *dst) const {
	const unsigned int pos =
		dealii::CUDAWrappers::local_q_point_id<dim, double>(cell,
															gpu_data,
															n_dofs_1d,
															n_q_points);

	dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
		fe_eval(cell, gpu_data, shared_data);
	fe_eval.read_dof_values(src);
	fe_eval.evaluate(true, true);
	fe_eval.apply_for_each_quad_point(
		HelmholtzOperatorQuad<dim, fe_degree>(coef[pos]));
	fe_eval.integrate(true, true);
	fe_eval.distribute_local_to_global(dst);
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::vmult(
	dealii::LinearAlgebra::distributed::Vector<double,
											   dealii::MemorySpace::CUDA> &Ax,
	dealii::LinearAlgebra::distributed::Vector<double,
											   dealii::MemorySpace::CUDA> &x)
	const {
	constexpr int dim = 3;

	Ax = 0.;
	LocalHelmholtzOperator<dim, FEOrderElectro> helmholtz_operator(
		d_coef.get_values());
	d_dealiiMFdata.set_constrained_values(0., x);
	d_dealiiMFdata.cell_loop(helmholtz_operator, x, Ax);
	d_dealiiMFdata.copy_constrained_values(x, Ax);
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::setupDealii() {
	constexpr int dim = 3;

	const dealii::DoFHandler<dim> &dofHandler =
		d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

	dealii::MappingQGeneric<dim> mapping(FEOrderElectro);

	typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
		additional_data(
			dealii::CUDAWrappers::MatrixFree<dim, double>::parallel_in_elem,
			dealii::update_values | dealii::update_gradients |
				dealii::update_JxW_values | dealii::update_quadrature_points,
			false,
			true);

	const dealii::QGauss<1> quad(FEOrderElectro + 1);
	d_dealiiMFdata.reinit(
		mapping, dofHandler, *d_constraintMatrixPtr, quad, additional_data);

	const unsigned int n_owned_cells =
		dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(
			&dofHandler.get_triangulation())
			->n_locally_owned_active_cells();

	d_coef.reinit(dealii::Utilities::pow(FEOrderElectro + 1, dim) *
				  n_owned_cells);

	const VaryingCoefficientFunctor<dim, FEOrderElectro> functor(
		d_coef.get_values());
	d_dealiiMFdata.evaluate_coefficients(functor);
}
