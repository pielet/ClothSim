#include "SparseMatrix.h"
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <vector>

namespace cloth
{
	template <typename T>
	__global__ void prefix_sum_kernel(T* values, T* block_sum)
	{
		int tid = threadIdx.x;
		T* start_values = values + 2 * blockIdx.x * blockDim.x;

		extern __shared__ T tmp[];

		tmp[2 * tid] = start_values[2 * tid];
		tmp[2 * tid + 1] = start_values[2 * tid + 1];
		__syncthreads();

		// bottom -> top
		int offset = 1;
		for (int i = blockDim.x; i > 0; i /= 2)
		{
			if (tid < i)
			{
				tmp[offset * 2 * tid] += tmp[offset * (2 * tid + 1)];
			}
			__syncthreads();
			offset *= 2;
		}

		// set zero
		if (tid == 0) tmp[tid] = 0;
		__syncthreads();

		// top -> bottom
		T swap;
		for (int i = 1; i < blockDim.x; i *= 2)
		{
			offset /= 2;
			if (tid < i)
			{
				swap = tmp[offset * 2 * tid];
				tmp[offset * 2 * tid] += tmp[offset * (2 * tid + 1)];
				tmp[offset * (2 * tid + 1)] = swap;
			}
			__syncthreads();
		}

		if (block_sum && tid == blockDim.x - 1)
			block_sum[blockIdx.x] = tmp[2 * tid + 1] + start_values[2 * tid + 1];

		start_values[2 * tid] = tmp[2 * tid];
		start_values[2 * tid + 1] = tmp[2 * tid + 1];
	}

	template <typename T>
	__global__ void add_block_sum_kernel(T* value, T* block_sum)
	{
		value[blockIdx.x * blockDim.x + threadIdx.x] += block_sum[blockIdx.x];
	}

	/// exclusive scan - adopted from Gems Chapter 39
	/// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
	template <typename T>
	void prefix_sum(int num_block, int block_size, T* values)
	{
		if (num_block == 0) return;

		if (num_block == 1)
		{
			prefix_sum_kernel <<< num_block, block_size / 2, block_size * sizeof(T) >>> (values, NULL);
		}
		else
		{
			T* block_sum;
			int new_num_block = (num_block + block_size - 1) / block_size;
			cudaMalloc((void**)&block_sum, new_num_block * block_size * sizeof(T));
			cudaMemset(block_sum, 0, new_num_block * block_size * sizeof(T));

			prefix_sum_kernel <<< num_block, block_size / 2, block_size * sizeof(T) >>> (values, block_sum);
			prefix_sum(new_num_block, block_size, block_sum);
			add_block_sum_kernel <<< num_block, block_size >>> (value, block_sum);

			cudaFree(block_sum);
		}
	}

	// sort an array per thread
	template <typename T>
	__device__ int merge_sort(T* arr, T* merged, int left, int right)
	{
		if (left == right - 1) return right;

		int mid = (left + right) / 2;
		int l_end = merge_sort(arr, merged, left, mid);
		int r_end = merge_sort(arr, merged, mid, right);

		int i = left, j = mid, k = left;
		do {
			while (i < l_end && arr[i] < arr[j]) merged[k++] = arr[i++];
			while (i < l_end && j < r_end && arr[i] > arr[j]) merged[k++] = arr[j++];
			while (i < l_end && j < r_end && arr[i] == arr[j])
			{
				merged[k++] = arr[i++];
				++j;
			}
		} while (i < l_end && j < r_end);
		while (i < l_end) merged[k++] = arr[i++];
		while (j < r_end) merged[k++] = arr[j++];

		for (i = left; i < k; ++i) arr[i] = merged[i];

		return k;
	}

	//! Assume blockDim.x is the exponential of 2
	template <typename T>
	__device__ void exclusive_scan(T* values)
	{
		int tid = threadIdx.x;

		// bottom -> top
		int offset = 1;
		for (int i = blockDim.x / 2; i > 0; i >>= 1)
		{
			if (tid < i)
			{
				values[offset * (2 * tid + 2) - 1] += values[offset * (2 * tid + 1) - 1];
			}
			__syncthreads();
			offset <<= 1;
		}

		// set zero
		if (tid == blockDim.x - 1) values[tid] = 0;
		__syncthreads();

		// top -> bottom
		T swap;
		for (int i = 1; i < blockDim.x; i <<= 1)
		{
			offset >>= 1;
			if (tid < i)
			{
				swap = values[offset * (2 * tid + 2) - 1];
				values[offset * (2 * tid + 2) - 1] += values[offset * (2 * tid + 1) - 1];
				values[offset * (2 * tid + 1) - 1] = swap;
			}
			__syncthreads();
		}
	}

	//! radix sort
	//! reference: http://www.compsci.hunter.cuny.edu/~sweiss/course_materials/csci360/lecture_notes/radix_sort_cuda.cc
	__global__ void mergeSortKernel(int n_row, int* row_ptr, int* col_idx, int* new_row_ptr)
	{
		int bid = blockIdx.x;
		int tid = threadIdx.x;

		int start = row_ptr[bid], end = row_ptr[bid + 1];
		int size = end - start;

		extern __shared__ int buffer[];
		int* one_bits = &buffer[blockDim.x];

		one_bits[tid] = 0;
		if (tid < size) buffer[tid] = col_idx[start + tid];
		__syncthreads();

		// radix sort
		int value, bit;
		int num_ones;
		for (int i = 0; i < 32; ++i) // LSB -> MSB
		{
			value = buffer[tid];
			bit = (value >> i) & 1;
			if (tid < size) one_bits[tid] = bit;
			__syncthreads();
			num_ones = one_bits[size - 1];
			__syncthreads();
			exclusive_scan(one_bits);
			num_ones += one_bits[size - 1];
			
			if (tid < size)
			{
				if (!bit) // 0
					buffer[tid - one_bits[tid]] = value;
				else // 1
					buffer[size - num_ones + one_bits[tid]] = value;
			}
			__syncthreads();
		}

		// merge
		bit = 0;
		if (tid == 0) bit = 1;
		if (tid > 0 && tid < size && buffer[tid] != buffer[tid - 1]) bit = 1;
		one_bits[tid] = bit;
		__syncthreads();

		num_ones = one_bits[size - 1];
		__syncthreads();
		exclusive_scan(one_bits);
		num_ones += one_bits[size - 1];

		// output
		if (bit) col_idx[start + one_bits[tid]] = buffer[tid];
		if (tid < 3) 
		{
			new_row_ptr[3 * blockIdx.x + tid] = 3 * num_ones;
		}
	}

	__global__ void countFaceIndicesKernel(int n_face, const FaceIdx* face_idx, int* row_ptr, FaceIdx* start_idx)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_face) return;
		
		for (int j = 0; j < 3; ++j)
		{
			start_idx[i](j) = atomicAdd(&row_ptr[face_idx[i](j)], 3);
		}
	}

	__global__ void countEdgeIndicesKernel(int n_edge, const EdgeIdx* edge_idx, int* row_ptr, EdgeIdx* start_idx)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_edge) return;

		for (int j = 0; j < 4; ++j)
		{
			start_idx[i](j) = atomicAdd(&row_ptr[edge_idx[i](j)], 4);
		}
	}

	__global__ void fillFaceIndicesKernel(int n_face, const FaceIdx* faces, const FaceIdx* start_indices, const int* row_ptr, int* col_idx)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_face) return;

		FaceIdx face_idx = faces[i], start_idx = start_indices[i];
#pragma unroll
		for (int j = 0; j < 3; ++j)
		{
#pragma unroll
			for (int k = 0; k < 3; ++k)
			{
				*(col_idx + row_ptr[face_idx(j)] + start_idx(j) + k) = face_idx(k);
			}
		}
	}

	__global__ void fillEdgeIndicesKernel(int n_edge, const EdgeIdx* edges, const EdgeIdx* start_indices, const int* row_ptr, int* col_idx)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_edge) return;

		EdgeIdx edge_idx = edges[i], start_idx = start_indices[i];
#pragma unroll
		for (int j = 0; j < 4; ++j)
		{
#pragma unroll
			for (int k = 0; k < 4; ++k)
			{
				*(col_idx + row_ptr[edge_idx(j)] + start_idx(j) + k) = edge_idx(k);
			}
		}
	}

	__global__ void adjustColIdxKernel(int n, const int* row_ptr, const int* new_row_ptr, const int* col_idx, int* new_col_idx)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		int old_start = row_ptr[i / 3];
		int new_start = new_row_ptr[i];
		int length = (new_row_ptr[i + 1] - new_start) / 3;

		for (int j = 0; j < length; ++j)
		{
#pragma unroll
			for (int k = 0; k < 3; ++k)
				new_col_idx[new_start + 3 * j + k] = 3 * col_idx[old_start + j] + k;
		}
	}

	__global__ void getDiagonalKernel(int n, const int* row_ptr, const int* col_idx, int* diag_idx)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		int idx;
		for (idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx)
		{
			if (col_idx[idx] == i) break;
		}
		diag_idx[i] = idx;
	}

	SparseMatrix::SparseMatrix():
		m_n(0), m_nnz(0), m_value(NULL), m_row_ptr(NULL), m_col_idx(NULL), m_diagonal_idx(NULL)
	{}

	SparseMatrix::~SparseMatrix()
	{
		if (m_value) cudaFree(m_value);
		if (m_row_ptr) cudaFree(m_row_ptr);
		if (m_col_idx) cudaFree(m_col_idx);
		if (m_diagonal_idx) cudaFree(m_diagonal_idx);
	}

	void SparseMatrix::initialize(int n_node, int n_face, int n_edge, const FaceIdx* faces, const EdgeIdx* edges)
	{
		m_n = 3 * n_node;

		// 1. Count number of indices and start indices
		int* row_ptr;
		cudaMalloc((void**)&row_ptr, (n_node + 1) * sizeof(int));
		cudaMemset(row_ptr, 0, (n_node + 1) * sizeof(int));

		FaceIdx* face_start_idx;
		EdgeIdx* edge_start_idx;
		cudaMalloc((void**)&face_start_idx, n_face * sizeof(FaceIdx));
		cudaMalloc((void**)&edge_start_idx, n_edge * sizeof(EdgeIdx));

		countFaceIndicesKernel <<< get_block_num(n_face), g_block_dim >>> (n_face, faces, row_ptr, face_start_idx);
		countEdgeIndicesKernel <<< get_block_num(n_edge), g_block_dim >>> (n_edge, edges, row_ptr, edge_start_idx);

		//std::vector<int> test;
		//test.resize(n_node + 1);
		//cudaMemcpy(test.data(), row_ptr, test.size() * sizeof(int), cudaMemcpyDeviceToHost);
		//std::cout << '\n';
		//for (int i = 0; i < test.size(); ++i)
		//	std::cout << test[i] << ' ';

		//prefix_sum(get_block_num(n_node + 1), g_block_dim, m_row_ptr);
		int* max_row_dptr = thrust::max_element(thrust::device, row_ptr, row_ptr + n_node);
		int max_row;
		cudaMemcpy(&max_row, max_row_dptr, sizeof(int), cudaMemcpyDeviceToHost);
		thrust::exclusive_scan(thrust::device, row_ptr, row_ptr + n_node + 1, row_ptr);

		// 2. Fill col_idx
		cudaMemcpy(&m_nnz, &row_ptr[n_node], sizeof(int), cudaMemcpyDeviceToHost);	// not the true nnz yet
		assert(m_nnz == 9 * n_face + 16 * n_edge);
		int* col_idx;
		cudaMalloc((void**)&col_idx, m_nnz * sizeof(int));

		fillFaceIndicesKernel <<< get_block_num(n_face), g_block_dim >>> (n_face, faces, face_start_idx, row_ptr, col_idx);
		fillEdgeIndicesKernel <<< get_block_num(n_edge), g_block_dim >>> (n_edge, edges, edge_start_idx, row_ptr, col_idx);

		// 3. Merge sort
		cudaMalloc((void**)&m_row_ptr, (m_n + 1) * sizeof(int));
		int n_threads = 1;
		while (n_threads < max_row) n_threads <<= 2;
		mergeSortKernel <<< n_node, n_threads, 2 * n_threads * sizeof(int) >>> (n_node, row_ptr, col_idx, m_row_ptr);

		// 4. Adjust row_ptr and compact col_idx
		thrust::exclusive_scan(thrust::device, m_row_ptr, m_row_ptr + m_n + 1, m_row_ptr);
		cudaMemcpy(&m_nnz, &m_row_ptr[m_n], sizeof(int), cudaMemcpyDeviceToHost);
		cudaMalloc((void**)&m_col_idx, m_nnz * sizeof(int));
		adjustColIdxKernel <<< get_block_num(m_n), g_block_dim >>> (m_n, row_ptr, m_row_ptr, col_idx, m_col_idx);

		// 5. Alocate data
		cudaMalloc((void**)&m_value, m_nnz * sizeof(Scalar));
		cudaMalloc((void**)&m_diagonal_idx, m_n * sizeof(int));
		getDiagonalKernel <<< get_block_num(m_n), g_block_dim >>> (m_n, m_row_ptr, m_col_idx, m_diagonal_idx);

		// 6. Clean up
		cudaFree(row_ptr);
		cudaFree(col_idx);
		cudaFree(face_start_idx);
		cudaFree(edge_start_idx);
	}

	void SparseMatrix::initialize(SparseMatrix& other)
	{
		m_n = other.m_n;
		m_nnz = other.m_nnz;
		m_row_ptr = other.m_row_ptr;
		m_col_idx = other.m_col_idx;
		m_diagonal_idx = other.m_diagonal_idx;
		cudaMalloc((void**)&m_value, m_nnz * sizeof(Scalar));
		cudaMemset(m_value, 0, m_nnz * sizeof(Scalar));
	}

	void SparseMatrix::assign(SparseMatrix& other)
	{
		cudaMemcpy(m_value, other.m_value, m_nnz * sizeof(Scalar), cudaMemcpyDeviceToDevice);
	}

	__global__ void addInDiagonalKernel(int n, const int* diag_idx, Scalar* value, Scalar a)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		value[diag_idx[i]] += a;
	}

	void SparseMatrix::addInDiagonal(Scalar a)
	{
		addInDiagonalKernel <<< get_block_num(m_n), g_block_dim >>> (m_n, m_diagonal_idx, m_value, a);
	}

	__global__ void addInDiagonalKernel(int n, Scalar a, const int* diag_idx, const Scalar* M, Scalar* value)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		for (int j = 0; j < 3; ++j)
		{
			value[diag_idx[3 * i + j]] += a * M[i];
		}
	}

	void SparseMatrix::addInDiagonal(const Scalar* M, Scalar a)
	{
		addInDiagonalKernel <<< get_block_num(m_n / 3), g_block_dim >>> (m_n / 3, a, m_diagonal_idx, M, m_value);
	}

	__global__ void invDiagonalKernel(int n, const int* diag_idx, const Scalar* value, Scalar* out)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		out[i] = 1 / value[diag_idx[i]];
	}

	void SparseMatrix::invDiagonal(Scalar* out)
	{
		invDiagonalKernel <<< get_block_num(m_n), g_block_dim >>> (m_n, m_diagonal_idx, m_value, out);
	}

	void SparseMatrix::setZero()
	{
		cudaMemset(m_value, 0, m_nnz * sizeof(Scalar));
	}

	CUDA_CALLABLE_MEMBER Scalar* SparseMatrix::getValue()
	{
		return m_value;
	}

	CUDA_CALLABLE_MEMBER int* SparseMatrix::getRowPtr()
	{
		return m_row_ptr;
	}

	CUDA_CALLABLE_MEMBER int* SparseMatrix::getColIdx()
	{
		return m_col_idx;
	}

	CUDA_CALLABLE_MEMBER int* SparseMatrix::getDiagonalIdx()
	{
		return m_diagonal_idx;
	}

	SparseMatrixWrapper::SparseMatrixWrapper(SparseMatrix& A):
		m_value(A.getValue()), m_row_ptr(A.getRowPtr()), m_col_idx(A.getColIdx()), m_diagonal_idx(A.getDiagonalIdx())
	{}

	CUDA_MEMBER void SparseMatrixWrapper::atomicAddIdentity(int row, int col, Scalar value)
	{
		int i;
		for (i = m_row_ptr[3 * row]; i < m_row_ptr[3 * row + 1]; ++i)
		{
			if (m_col_idx[i] == 3 * col) break;
		}
		i -= m_row_ptr[3 * row];

#pragma unroll
		for (int j = 0; j < 3; ++j)
		{
			atomicAdd(&m_value[m_row_ptr[3 * row + j] + i + j], value);
		}
	}

	CUDA_MEMBER void SparseMatrixWrapper::atomicAddIdentity(int i, Scalar value)
	{
#pragma unroll
		for (int j = 0; j < 3; ++j)
		{
			atomicAdd(&m_value[m_diagonal_idx[3 * i + j]], value);
		}
	}

	CUDA_MEMBER void SparseMatrixWrapper::atomicAddBlock(int bi, int bj, const Mat<Scalar, 3, 3>& mat)
	{
		int row = 3 * bi;
		int col;
		for (col = m_row_ptr[row]; col < m_row_ptr[row + 1]; ++col)
		{
			if (m_col_idx[col] == 3 * bj) break;
		}
		col -= m_row_ptr[row];

#pragma unroll
		for (int i = 0; i < 3; ++i) 
#pragma unroll
			for (int j = 0; j < 3; ++j)
		{
			atomicAdd(&m_value[m_row_ptr[row + i] + col + j], mat(i, j));
		}
	}
}