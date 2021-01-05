#include "Constraints.h"
#include <cuda_runtime.h>
#include "ClothSim.h"

namespace cloth
{
	StretchingConstraints::StretchingConstraints():
		m_indices(NULL), m_areas(NULL), m_Dm(NULL), m_Dm_inv(NULL), m_num_faces(0)
	{}

	StretchingConstraints::~StretchingConstraints()
	{
		if (m_indices) cudaFree(m_indices);
		if (m_areas) cudaFree(m_areas);
		if (m_Dm) cudaFree(m_Dm);
		if (m_Dm_inv) cudaFree(m_Dm_inv);
	}

	const FaceIdx* StretchingConstraints::getIndices() const
	{
		return m_indices;
	}

	__global__ void stretchingInitializeKernel(int n_faces, int offset, const FaceIdx* indices, const Vec3x* x, const Vec2x* uv, Scalar* areas, Mat2x* Dms, Mat2x* Dm_invs)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_faces) return;

		FaceIdx idx = indices[i] + offset;
		Vec3x p0 = x[idx(0)], p1 = x[idx(1)], p2 = x[idx(2)];
		Vec2x u0 = uv[idx(0)], u1 = uv[idx(1)], u2 = uv[idx(2)];

		areas[i] = 0.5f * ((p1 - p0).cross(p2 - p0).norm());

		Mat2x Dm;
		Dm.setCol(0, u1 - u0);
		Dm.setCol(1, u2 - u0);
		Dms[i] = Dm;
		Dm_invs[i] = Dm.inverse();
	}

	void StretchingConstraints::initialize(int n_faces, int offset, const MaterialParameters* material, const FaceIdx* indices, const Vec3x* x, const Vec2x* uv)
	{
		m_num_faces = n_faces;
		m_offset = offset;

		m_type = material->m_type;
		m_mu = material->m_thickness * material->m_youngs_modulus / (1 + material->m_possion_ratio) / 2.f;
		m_lambda = material->m_thickness * material->m_youngs_modulus * material->m_possion_ratio /
			(1 + material->m_possion_ratio) / (1 - 2 * material->m_possion_ratio);

		cudaMalloc((void**)&m_indices, n_faces * sizeof(FaceIdx));
		cudaMalloc((void**)&m_areas, n_faces * sizeof(Scalar));
		cudaMalloc((void**)&m_Dm, n_faces * sizeof(Mat2x));
		cudaMalloc((void**)&m_Dm_inv, n_faces * sizeof(Mat2x));

		cudaMemcpy(m_indices, indices, n_faces * sizeof(FaceIdx), cudaMemcpyHostToDevice);
		stretchingInitializeKernel <<< get_block_num(n_faces), g_block_dim >>> (n_faces, offset, m_indices, x, uv, m_areas, m_Dm, m_Dm_inv);
	}

	void StretchingConstraints::computeGradiant(Scalar* gradient)
	{

	}

	BendingConstraints::BendingConstraints():
		m_indices(NULL), m_stiffness(NULL), m_K(NULL), m_num_edges(0)
	{}

	BendingConstraints::~BendingConstraints()
	{
		if (m_indices) cudaFree(m_indices);
		if (m_stiffness) cudaFree(m_stiffness);
		if (m_K) cudaFree(m_K);
	}

	const EdgeIdx* BendingConstraints::getIndices() const
	{
		return m_indices;
	}

	__global__ void bendingInitializeKernel(int n_edges, int offset, Scalar scalar, const EdgeIdx* indices, const Vec3x* x, Scalar* stiffness, Vec4x* Ks)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_edges) return;

		EdgeIdx idx = indices[i] + offset;
		Vec3x p0 = x[idx(0)], p1 = x[idx(1)], p2 = x[idx(2)], p3 = x[idx(3)];

		Scalar l2 = (p1 - p0).squareNorm();
		Scalar a0 = (p2 - p0).cross(p1 - p0).norm(), 
			   a1 = (p3 - p0).cross(p1 - p0).norm();
		Scalar tri0 = l2 / a0, tri1 = l2 / a1;

		Scalar alpha0 = (p2 - p0).dot(p1 - p0) / l2,
			alpha1 = (p3 - p0).dot(p1 - p0) / l2,
			beta0 = 1 - alpha0, beta1 = 1 - alpha1;

		Ks[i](0) = beta0 * tri0 + beta1 * tri1;
		Ks[i](1) = alpha0 * tri0 + alpha1 * tri1;
		Ks[i](2) = -tri0;
		Ks[i](3) = -tri1;

		stiffness[i] = 6 / (a0 + a1) * scalar;
	}

	void BendingConstraints::intialize(int n_edges, int offset, const MaterialParameters* material, const EdgeIdx* indices, const Vec3x* x)
	{
		m_num_edges = n_edges;
		m_offset = offset;

		cudaMalloc((void**)&m_indices, n_edges * sizeof(EdgeIdx));
		cudaMalloc((void**)&m_stiffness, n_edges * sizeof(Scalar));
		cudaMalloc((void**)&m_K, n_edges * sizeof(Vec4x));

		cudaMemcpy(m_indices, indices, n_edges * sizeof(EdgeIdx), cudaMemcpyHostToDevice);
		bendingInitializeKernel <<< get_block_num(n_edges), g_block_dim >>> (n_edges, offset, material->m_bending_stiffness, m_indices, x, m_stiffness, m_K);
	}

	__global__ void precomputeKernel(int n_edges, const EdgeIdx* indices, const Scalar* scalars)
	{

	}

	void BendingConstraints::precompute(SparseMat3x& sparse_mat)
	{

	}
}