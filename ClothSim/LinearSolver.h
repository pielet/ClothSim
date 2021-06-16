#ifndef LINEAR_SOLVER
#define LINEAR_SOLVER

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "Utils/Cublas.h"
#include "SparseMatrix.h"

namespace cloth
{
	struct CudaMvWrapper
	{
		cusparseHandle_t m_handle;

		cusparseSpMatDescr_t m_matA;
		cusparseDnVecDescr_t m_vecx;
		cusparseDnVecDescr_t m_vecAx;

		void* d_buffer;

		CudaMvWrapper();
		~CudaMvWrapper();
		void initialize(SparseMatrix* mat, Scalar* x, Scalar* Ax);
		void mv();
	};

	class PreconditionerBase;

	class LinearSolver
	{
	public:
		enum PrecondT { NoPreconditionner, Diagnal, Factorization };

		LinearSolver();
		~LinearSolver();

		//! Initialize matrix descriptor and preconditioner (for CG)
		void initialize(SparseMatrix* matrix, PrecondT pt = NoPreconditionner);

		//! Perform LLT pre-factorization
		void cholFactor(SparseMatrix* matrix);

		//! Perform solve phase of LLT (cholFactor should be called before)
		void cholSolve(const Scalar* b, Scalar* x);

		bool cholesky(const Scalar* b, Scalar* x);
		bool conjugateGradient(const Scalar* b, Scalar* x, int iters, Scalar err);

	private:
		int m_n;

		SparseMatrix* m_matrix;
		PreconditionerBase* m_precond;

		cublasHandle_t m_cublasHandle;
		cusolverSpHandle_t m_cusolverSpHandle;

		// Cholesky
		cusparseMatDescr_t m_descrA;
		csrcholInfo_t d_info;

		// CG
		CudaMvWrapper m_mv_caller;

		/* Device pointer */
		Scalar* d_r;
		Scalar* d_p;
		Scalar* d_z;
		Scalar* d_Ap;
		void* d_buffer;
	};

	// Preconditioner
	class PreconditionerBase
	{
	public:
		virtual ~PreconditionerBase() {};
		virtual bool analysis() = 0;
		virtual void solve(const Scalar* in, Scalar* out) = 0;
	};

	class DummyPreconditioner :public PreconditionerBase
	{
	public:
		DummyPreconditioner(int n) :m_n(n) {}
		virtual bool analysis() { return true; }
		virtual void solve(const Scalar* in, Scalar* out);
	private:
		int m_n;
	};

	class DiagnalPreconditioner :public PreconditionerBase
	{
	public:
		DiagnalPreconditioner(SparseMatrix* A);
		~DiagnalPreconditioner();
		virtual bool analysis();
		virtual void solve(const Scalar* in, Scalar* out);
	private:
		SparseMatrix* m_A;
		Scalar* m_invDiagA;
	};

	class FactorizationPreconditioner :public PreconditionerBase
	{
	public:
		FactorizationPreconditioner(SparseMatrix* A);
		~FactorizationPreconditioner();
		virtual bool analysis();
		virtual void solve(const Scalar* in, Scalar* out);
	private:
		SparseMatrix* m_A;

		Scalar* m_valsILU;
		Scalar* m_y;
		void* m_buffer;

		cusparseHandle_t m_cusparseHandle;

		cusparseMatDescr_t m_descrA;

		csrilu02Info_t m_infoILU;
		csrsv2Info_t m_infoL;
		csrsv2Info_t m_infoU;
		cusparseMatDescr_t m_descrL;
		cusparseMatDescr_t m_descrU;
	};

	template <typename ScalarT> struct CusparseCaller;
	template <typename ScalarT> struct CusolverCaller;
}

#endif // !LINEAR_SOLVER
