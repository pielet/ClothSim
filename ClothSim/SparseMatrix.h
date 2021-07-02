// Auther: Shiyang Jia (jsy0325@foxmail.com)
// Date: 12/28/2020
#pragma once

#include <cuda_runtime.h>
#include "Utils/MathDef.h"

namespace cloth
{
	//! CSR format sparse matrix
	class SparseMatrix
	{
	public:
		SparseMatrix();
		~SparseMatrix();
		
		//! Allocates and compactes memory
		void initialize(int n_node, int n_face, int n_edge, const FaceIdx* faces, const EdgeIdx* edges);
		
		//! After initialize, this row_ptr and col_idx point to the same address as others'
		void initialize(SparseMatrix& other);

		//! Copies value vector from [other] to [*this]
		void assign(SparseMatrix& other);

		//! Adds a scalar to diagonal elements
		void addInDiagonal(Scalar a);

		//! Adds diagonal matrix to block of *this with scaling a
		void addInDiagonal(const Scalar* M, Scalar a = 1.f);

		//! Gets inverse diagonal element
		void invDiagonal(Scalar* out);

		void setZero();

		//! Access
		int getn() const { return m_n; }
		int getnnz() const { return m_nnz; }

		CUDA_CALLABLE_MEMBER Scalar* getValue();
		CUDA_CALLABLE_MEMBER int* getRowPtr();
		CUDA_CALLABLE_MEMBER int* getColIdx();
		CUDA_CALLABLE_MEMBER int* getDiagonalIdx();

	private:
		int m_n; //< num_nodes
		int m_nnz; //< non-zero blocks

		//! BSR stuff
		Scalar* m_value;
		int* m_row_ptr;
		int* m_col_idx;

		int* m_diagonal_idx;
	};

	class SparseMatrixWrapper
	{
	public:
		SparseMatrixWrapper(SparseMatrix& A);

		//! Add [value] to the diagonal of (i, j)-th 3x3 block
		CUDA_MEMBER void atomicAddIdentity(int i, int j, Scalar value);
		//! Add [value] to the i-th diagonal block
		CUDA_MEMBER void atomicAddIdentity(int i, Scalar value);
		//! Add mat to (i, j)-th 3x3 block
		CUDA_MEMBER void atomicAddBlock(int i, int j, const Mat<Scalar, 3, 3>& mat);

	private:
		Scalar* m_value;
		int* m_row_ptr;
		int* m_col_idx;

		int* m_diagonal_idx;
	};
}
