#include "LinearSolver.h"

namespace cloth
{
	template <>
	struct CusparseCaller<double>
	{
		static void createCsr(cusparseSpMatDescr_t* mat, int n, int nnz, int* rowPtr, int* colIdx, double* values)
		{
			checkCudaErrors(cusparseCreateCsr(
				mat, n, n, nnz, rowPtr, colIdx, values,
				CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
		}

		static void createDnVec(cusparseDnVecDescr_t* vec, int n, double* values)
		{
			checkCudaErrors(cusparseCreateDnVec(vec, n, values, CUDA_R_64F));
		}

		static void mv_bufferSize(cusparseHandle_t handle, const double* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecAx,
			const double* beta, cusparseDnVecDescr_t vecx, size_t* bufferSize)
		{
			checkCudaErrors(cusparseSpMV_bufferSize(
				handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecAx, beta, vecx,
				CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, bufferSize));
		}

		static void mv(cusparseHandle_t handle, const double* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecx,
			const double* beta, cusparseDnVecDescr_t vecAx, void* buffer)
		{
			checkCudaErrors(cusparseSpMV(
				handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecx, beta, vecAx,
				CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, buffer));
		}

		static void ilu_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, double* values,
			const int* rowPtr, const int* colIdx, csrilu02Info_t info, int* bufferSize)
		{
			checkCudaErrors(cusparseDcsrilu02_bufferSize(handle, n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
		}

		static void ilu_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const double* values,
			const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
		{
			checkCudaErrors(cusparseDcsrilu02_analysis(handle, n, nnz, descr, values, rowPtr, colIdx, info,
				CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}

		static void ilu(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, double* values,
			const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
		{
			checkCudaErrors(cusparseDcsrilu02(handle, n, nnz, descr, values, rowPtr, colIdx, info,
				CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}

		static void sv2_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, double* values,
			const int* rowPtr, const int* colIdx, csrsv2Info_t info, int* bufferSize)
		{
			checkCudaErrors(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
		}

		static void sv2_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const double* values,
			const int* rowPtr, const int* colIdx, csrsv2Info_t info, void* buffer)
		{
			checkCudaErrors(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
				descr, values, rowPtr, colIdx, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}

		static void sv2_solve(cusparseHandle_t handle, int n, int nnz, const double* alpha, const cusparseMatDescr_t descr, const double* values,
			const int* rowPtr, const int* colIdx, csrsv2Info_t info, const double* x, double* y, void* buffer)
		{
			checkCudaErrors(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, alpha,
				descr, values, rowPtr, colIdx, info, x, y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}
	};

	template <>
	struct CusparseCaller<float>
	{
		static void createCsr(cusparseSpMatDescr_t* mat, int n, int nnz, int* rowPtr, int* colIdx, float* values)
		{
			checkCudaErrors(cusparseCreateCsr(
				mat, n, n, nnz, rowPtr, colIdx, values,
				CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
		}

		static void createDnVec(cusparseDnVecDescr_t* vec, int n, float* values)
		{
			checkCudaErrors(cusparseCreateDnVec(vec, n, values, CUDA_R_32F));
		}

		static void mv_bufferSize(cusparseHandle_t handle, const float* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecAx,
			const float* beta, cusparseDnVecDescr_t vecx, size_t* bufferSize)
		{
			checkCudaErrors(cusparseSpMV_bufferSize(
				handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecAx, &beta, vecx,
				CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, bufferSize));
		}

		static void mv(cusparseHandle_t handle, const float* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecx,
			const float* beta, cusparseDnVecDescr_t vecAx, void* buffer)
		{
			checkCudaErrors(cusparseSpMV(
				handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA,
				vecx, beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, buffer));
		}

		static void ilu_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, float* values,
			const int* rowPtr, const int* colIdx, csrilu02Info_t info, int* bufferSize)
		{
			checkCudaErrors(cusparseScsrilu02_bufferSize(handle, n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
		}

		static void ilu_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const float* values,
			const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
		{
			checkCudaErrors(cusparseScsrilu02_analysis(handle, n, nnz, descr, values, rowPtr, colIdx, info,
				CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}

		static void ilu(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, float* values,
			const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
		{
			checkCudaErrors(cusparseScsrilu02(handle, n, nnz, descr, values, rowPtr, colIdx, info,
				CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}

		static void sv2_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, float* values,
			const int* rowPtr, const int* colIdx, csrsv2Info_t info, int* bufferSize)
		{
			checkCudaErrors(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
		}

		static void sv2_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const float* values,
			const int* rowPtr, const int* colIdx, csrsv2Info_t info, void* buffer)
		{
			checkCudaErrors(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
				descr, values, rowPtr, colIdx, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}

		static void sv2_solve(cusparseHandle_t handle, int n, int nnz, const float* alpha, const cusparseMatDescr_t descr, const float* values,
			const int* rowPtr, const int* colIdx, csrsv2Info_t info, const float* x, float* y, void* buffer)
		{
			checkCudaErrors(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, alpha,
				descr, values, rowPtr, colIdx, info, x, y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
		}
	};

	template <>
	struct CusolverCaller<double>
	{
		static void cholesky(cusolverSpHandle_t handle, int n, int nnz, cusparseMatDescr_t descrA, const double* values, const int* rowPtr, const int* colIdx, const double* b, double* x, int* singularity)
		{
			cusolverSpDcsrlsvchol(handle, n, nnz, descrA, values, rowPtr, colIdx, b, EPS, 2, x, singularity);  // symamd
			//Eigen::VecXx test(nnz);
			//cudaMemcpy(test.data(), values, nnz * sizeof(double), cudaMemcpyDeviceToHost);
			//std::cout << test << "\n\n\n";
			//test.resize(n);
			//cudaMemcpy(test.data(), b, n * sizeof(double), cudaMemcpyDeviceToHost);
			//std::cout << test;
		}
	};

	template <>
	struct CusolverCaller<float>
	{
		static void cholesky(cusolverSpHandle_t handle, int n, int nnz, cusparseMatDescr_t descrA, const float* values, const int* rowPtr, const int* colIdx, const float* b, float* x, int* singularity)
		{
			cusolverSpScsrlsvchol(handle, n, nnz, descrA, values, rowPtr, colIdx, b, EPS, 2, x, singularity);  // symamd
		}
	};

	CudaMvWrapper::CudaMvWrapper(): 
		d_buffer(NULL), m_handle(NULL), m_matA(NULL), m_vecx(NULL), m_vecAx(NULL)
	{}

	CudaMvWrapper::~CudaMvWrapper()
	{
		if (d_buffer) cudaFree(d_buffer);

		if (m_handle) cusparseDestroy(m_handle);

		if (m_matA) cusparseDestroySpMat(m_matA);
		if (m_vecx) cusparseDestroyDnVec(m_vecx);
		if (m_vecAx) cusparseDestroyDnVec(m_vecAx);
	}

	void CudaMvWrapper::initialize(SparseMatrix* A, Scalar* x, Scalar* Ax)
	{
		cusparseCreate(&m_handle);

		CusparseCaller<Scalar>::createCsr(&m_matA, A->getn(), A->getnnz(), A->getRowPtr(), A->getColIdx(), A->getValue());
		CusparseCaller<Scalar>::createDnVec(&m_vecx, A->getn(), x);
		CusparseCaller<Scalar>::createDnVec(&m_vecAx, A->getn(), Ax);

		size_t buffer_size = 0;
		Scalar one = 1, zero = 0;
		CusparseCaller<Scalar>::mv_bufferSize(m_handle, &one, m_matA, m_vecAx, &zero, m_vecx, &buffer_size);

		cudaMalloc((void**)&d_buffer, buffer_size);
	}

	void CudaMvWrapper::mv()
	{
		Scalar one = 1, zero = 0;
		CusparseCaller<Scalar>::mv(m_handle, &one, m_matA, m_vecx, &zero, m_vecAx, d_buffer);
	}

	LinearSolver::LinearSolver():
		d_r(NULL), d_p(NULL), d_z(NULL), d_Ap(NULL), m_precond(NULL)
	{}

	LinearSolver::~LinearSolver()
	{
		if (d_r) cudaFree(d_r);
		if (d_p) cudaFree(d_p);
		if (d_z) cudaFree(d_z);
		if (d_Ap) cudaFree(d_Ap);

		if (m_cublasHandle) cublasDestroy(m_cublasHandle);
		if (m_cusolverSpHandle) cusolverSpDestroy(m_cusolverSpHandle);
		if (m_descrA) cusparseDestroyMatDescr(m_descrA);

		if (m_precond) delete m_precond;
	}

	void LinearSolver::initialize(SparseMatrix* mat, PrecondT pt)
	{
		m_matrix = mat;
		m_n = mat->getn();

		switch (pt)
		{
		case LinearSolver::NoPreconditionner:
			m_precond = new DummyPreconditioner(m_n);
			break;
		case LinearSolver::Diagnal:
			m_precond = new DiagnalPreconditioner(mat);
			break;
		case LinearSolver::Factorization:
			m_precond = new FactorizationPreconditioner(mat);
			break;
		}

		cublasCreate(&m_cublasHandle);
		cusolverSpCreate(&m_cusolverSpHandle);

		cusparseCreateMatDescr(&m_descrA);
		cusparseSetMatType(m_descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(m_descrA, CUSPARSE_INDEX_BASE_ZERO);

		cudaMalloc((void**)&d_r, m_n * sizeof(Scalar));
		cudaMalloc((void**)&d_p, m_n * sizeof(Scalar));
		cudaMalloc((void**)&d_z, m_n * sizeof(Scalar));
		cudaMalloc((void**)&d_Ap, m_n * sizeof(Scalar));

		m_mv_caller.initialize(mat, d_p, d_Ap);
	}

	bool LinearSolver::cholesky(const Scalar* b, Scalar* x)
	{
		int singularity;
		CusolverCaller<Scalar>::cholesky(m_cusolverSpHandle, m_n, m_matrix->getnnz(), m_descrA, m_matrix->getValue(),
			m_matrix->getRowPtr(), m_matrix->getColIdx(), b, x, &singularity);
		return singularity < 0;
	}

	bool LinearSolver::conjugateGradient(const Scalar* b, Scalar* x)
	{
		Scalar one = 1.0, neg_one = -1.0;
		Scalar res, bnorm, alpha, beta;
		Scalar pAp, rz, old_rz;

		int nnz = m_matrix->getnnz();
		Scalar* values = m_matrix->getValue();
		const int* rowPtr = m_matrix->getRowPtr();
		const int* colIdx = m_matrix->getColIdx();

		// Perform analysis for ILU
		bool status = m_precond->analysis();
		if (!status)
		{
			//std::cerr << "Preconditioner analysis failed. EXIT." << std::endl;
			exit(-1);
		}

		// r0 = b - Ax
		CublasCaller<Scalar>::copy(m_cublasHandle, m_n, x, d_p);
		CublasCaller<Scalar>::copy(m_cublasHandle, m_n, b, d_r);

		m_mv_caller.mv();
		CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &neg_one, d_Ap, d_r);

		CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_r, &res);
		CublasCaller<Scalar>::dot(m_cublasHandle, m_n, b, b, &bnorm);

		//if (res / bnorm < eps) return;

		m_precond->solve(d_r, d_z);
		CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_z, d_p);
		CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_z, &rz);

		Eigen::VecXx test(m_n);

		int k = 0;
		for (k; k < m_n; ++k)
		{
			m_mv_caller.mv();
			CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_p, d_Ap, &pAp);
			alpha = rz / pAp;

			CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &alpha, d_p, x);
			alpha = -alpha;
			CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &alpha, d_Ap, d_r);
			old_rz = rz;

			CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_r, &res);
			//std::cout << "\t iter: " << k << " rTr: " << res << std::endl;

			//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_p, d_Ap);
			//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, x, d_p);
			//CusparseCaller<Scalar>::mv(m_cusparseHandle, &one, m_matA, m_vecp, &zero, m_vecAp, d_buffer);
			//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_r, d_z);
			//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_b, d_r);
			//CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &neg_one, d_Ap, d_r);
			//CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_r, &res);
			//cudaMemcpy(test.data(), x, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
			//for (int i = 0; i < m_n; ++i)
			//	std::cout << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1) << test(i) << ' ';
			//std::cout << '\n';
			//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_Ap, d_p);
			//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_z, d_r);

			//if (res < eps * bnorm) break;

			m_precond->solve(d_r, d_z);
			CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_z, &rz);
			beta = rz / old_rz;
			CublasCaller<Scalar>::scal(m_cublasHandle, m_n, &beta, d_p);
			CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &one, d_z, d_p);
		}

		//std::cout << "Total CG iteration: " << k << " residual: " << res << std::endl;

		return true;
	}

	void DummyPreconditioner::solve(const Scalar* in, Scalar* out)
	{
		cudaMemcpy(out, in, m_n * sizeof(Scalar), cudaMemcpyDeviceToDevice);
	}

	DiagnalPreconditioner::DiagnalPreconditioner(SparseMatrix* A) :m_A(A)
	{
		checkCudaErrors(cudaMalloc((void**)&m_invDiagA, A->getn() * sizeof(Scalar)));
	}

	DiagnalPreconditioner::~DiagnalPreconditioner()
	{
		cudaFree(m_invDiagA);
	}

	bool DiagnalPreconditioner::analysis()
	{
		m_A->invDiagonal(m_invDiagA);
		return true;
	}

	__global__ void cwiseMultiply(int n, const Scalar* a, const Scalar* b, Scalar* c)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		c[i] = a[i] * b[i];
	}

	void DiagnalPreconditioner::solve(const Scalar* in, Scalar* out)
	{
		cwiseMultiply <<< get_block_num(m_A->getn()), g_block_dim >>> (m_A->getn(), m_invDiagA, in, out);
	}

	FactorizationPreconditioner::FactorizationPreconditioner(SparseMatrix* A) :m_A(A)
	{
		checkCudaErrors(cusparseCreate(&m_cusparseHandle));

		checkCudaErrors(cusparseCreateMatDescr(&m_descrA));
		checkCudaErrors(cusparseSetMatType(m_descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatIndexBase(m_descrA, CUSPARSE_INDEX_BASE_ZERO));

		// Creates ILU info and triangular solve info
		checkCudaErrors(cusparseCreateCsrilu02Info(&m_infoILU));

		checkCudaErrors(cusparseCreateCsrsv2Info(&m_infoL));
		checkCudaErrors(cusparseCreateCsrsv2Info(&m_infoU));

		checkCudaErrors(cusparseCreateMatDescr(&m_descrL));
		checkCudaErrors(cusparseSetMatType(m_descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatIndexBase(m_descrL, CUSPARSE_INDEX_BASE_ZERO));
		checkCudaErrors(cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER));
		checkCudaErrors(cusparseSetMatDiagType(m_descrL, CUSPARSE_DIAG_TYPE_UNIT));

		checkCudaErrors(cusparseCreateMatDescr(&m_descrU));
		checkCudaErrors(cusparseSetMatType(m_descrU, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatIndexBase(m_descrU, CUSPARSE_INDEX_BASE_ZERO));
		checkCudaErrors(cusparseSetMatFillMode(m_descrU, CUSPARSE_FILL_MODE_UPPER));
		checkCudaErrors(cusparseSetMatDiagType(m_descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));

		cudaMalloc((void**)&m_valsILU, A->getnnz() * sizeof(Scalar));
		cudaMalloc((void**)&m_y, A->getn() * sizeof(Scalar));

		size_t bufferSize = 0;
		int tmp;

		CusparseCaller<Scalar>::ilu_bufferSize(m_cusparseHandle, A->getn(), A->getnnz(), m_descrA,
			A->getValue(), A->getRowPtr(), A->getColIdx(), m_infoILU, &tmp);
		if (tmp > bufferSize) bufferSize = tmp;

		CusparseCaller<Scalar>::sv2_bufferSize(m_cusparseHandle, A->getn(), A->getnnz(), m_descrL,
			A->getValue(), A->getRowPtr(), A->getColIdx(), m_infoL, &tmp);
		if (tmp > bufferSize) bufferSize = tmp;

		CusparseCaller<Scalar>::sv2_bufferSize(m_cusparseHandle, A->getn(), A->getnnz(), m_descrU,
			A->getValue(), A->getRowPtr(), A->getColIdx(), m_infoU, &tmp);
		if (tmp > bufferSize) bufferSize = tmp;

		checkCudaErrors(cudaMalloc(&m_buffer, bufferSize));
	}

	FactorizationPreconditioner::~FactorizationPreconditioner()
	{
		cusparseDestroy(m_cusparseHandle);
		cusparseDestroyMatDescr(m_descrA);

		cusparseDestroyCsrilu02Info(m_infoILU);
		cusparseDestroyCsrsv2Info(m_infoL);
		cusparseDestroyCsrsv2Info(m_infoU);
		cusparseDestroyMatDescr(m_descrL);
		cusparseDestroyMatDescr(m_descrU);

		cudaFree(m_valsILU);
		cudaFree(m_y);
		cudaFree(m_buffer);
	}

	bool FactorizationPreconditioner::analysis()
	{
		int n = m_A->getn(), nnz = m_A->getnnz();
		Scalar* values = m_A->getValue();
		const int* rowPtr = m_A->getRowPtr();
		const int* colIdx = m_A->getColIdx();

		int structural_zero, numerical_zero;

		// Perform analysis for ILU
		CusparseCaller<Scalar>::ilu_analysis(m_cusparseHandle, n, nnz, m_descrA, values, rowPtr, colIdx, m_infoILU, m_buffer);

		auto status = cusparseXcsrilu02_zeroPivot(m_cusparseHandle, m_infoILU, &structural_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
			printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
		}

		cudaMemcpy(m_valsILU, values, nnz * sizeof(Scalar), cudaMemcpyDeviceToDevice);
		CusparseCaller<Scalar>::ilu(m_cusparseHandle, n, nnz, m_descrA, m_valsILU, rowPtr, colIdx, m_infoILU, m_buffer);

		status = cusparseXcsrilu02_zeroPivot(m_cusparseHandle, m_infoILU, &numerical_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
			printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
		}

		// Perform analysis for trianguler
		CusparseCaller<Scalar>::sv2_analysis(m_cusparseHandle, n, nnz, m_descrL, m_valsILU, rowPtr, colIdx, m_infoL, m_buffer);
		CusparseCaller<Scalar>::sv2_analysis(m_cusparseHandle, n, nnz, m_descrU, m_valsILU, rowPtr, colIdx, m_infoU, m_buffer);

		return structural_zero < 0 && numerical_zero < 0;
	}

	void FactorizationPreconditioner::solve(const Scalar* in, Scalar* out)
	{
		Scalar one = 1.0;

		// out = U^-1 * L^-1 * in
		CusparseCaller<Scalar>::sv2_solve(m_cusparseHandle, m_A->getn(), m_A->getnnz(), &one, m_descrL,
			m_valsILU, m_A->getRowPtr(), m_A->getColIdx(), m_infoL, in, m_y, m_buffer);
		CusparseCaller<Scalar>::sv2_solve(m_cusparseHandle, m_A->getn(), m_A->getnnz(), &one, m_descrU,
			m_valsILU, m_A->getRowPtr(), m_A->getColIdx(), m_infoU, m_y, out, m_buffer);
	}
}