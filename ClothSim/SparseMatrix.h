// Auther: Shiyang Jia (jsy0325@foxmail.com)
// Date: 12/28/2020
#pragma once

#include <cuda_runtime.h>
#include "Utils/MathDef.h"

namespace cloth
{
	//template <typename ScalarType>
	//struct Sparse
	//{
	//	ScalarType* m_value;
	//	int* m_row_ptr;
	//	int* m_col_idx;
	//};

	//! BSR format sparse matrix
	template <typename ScalarType, int block_dim>
	class SparseMatrix
	{
	public:
		SparseMatrix();
		~SparseMatrix();
		
		void initialize(int n_node, int n_face, int n_edge, const FaceIdx* faces, const EdgeIdx* edges);
		void initialize(SparseMatrix<ScalarType, block_dim>& other);

		CUDA_CALLABLE_MEMBER ScalarType* getValue();
		CUDA_CALLABLE_MEMBER int* getRowPtr();
		CUDA_CALLABLE_MEMBER int* getColIdx();

	private:
		int m_n; //< num_nodes
		int m_nnz; //< non-zero blocks

		//! BSR stuff
		ScalarType* m_value;
		int* m_row_ptr;
		int* m_col_idx;
	};

	typedef SparseMatrix<Scalar, 3> SparseMat3x;
}
