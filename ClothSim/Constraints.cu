#include "Constraints.h"
#include <cuda_runtime.h>
#include "ClothSim.h"

#include <vector>
#include <iostream>

namespace cloth
{
	__device__ Vec3x atomicAddVec3x(Vec3x* addr, const Vec3x& value)
	{
		Vec3x old;

		old(0) = atomicAdd(&addr->value[0], value(0));
		old(1) = atomicAdd(&addr->value[1], value(1));
		old(2) = atomicAdd(&addr->value[2], value(2));

		return old;
	}

	/***************************** STRETCHING CONSTRAINTS **************************/
	StretchingConstraints::StretchingConstraints():
		m_indices(NULL), m_areas(NULL), m_Dm(NULL), m_Dm_inv(NULL), m_energy(NULL), m_handle(NULL), m_num_faces(0)
	{}

	StretchingConstraints::~StretchingConstraints()
	{
		if (m_indices) cudaFree(m_indices);
		if (m_areas) cudaFree(m_areas);
		if (m_Dm) cudaFree(m_Dm);
		if (m_Dm_inv) cudaFree(m_Dm_inv);
		if (m_energy) cudaFree(m_energy);

		if (m_handle) cublasDestroy(m_handle);
	}

	const FaceIdx* StretchingConstraints::getIndices() const
	{
		return m_indices;
	}

	__global__ void stretchingInitializeKernel(int n_faces, const FaceIdx* indices, const Vec3x* x, const Vec2x* uv, 
		Scalar* areas, Mat2x* Dms, Mat2x* Dm_invs, Scalar* mass)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_faces) return;

		FaceIdx idx = indices[i];
		Vec3x p0 = x[idx(0)], p1 = x[idx(1)], p2 = x[idx(2)];
		Vec2x u0 = uv[idx(0)], u1 = uv[idx(1)], u2 = uv[idx(2)];

		Scalar area = 0.5f * ((p1 - p0).cross(p2 - p0).norm());
		areas[i] = area;

		Mat2x Dm;
		Dm.setCol(0, u0 - u2);
		Dm.setCol(1, u1 - u2);
		Dms[i] = Dm;
		Dm_invs[i] = Dm.inverse();

		area /= 3;
		atomicAdd(&mass[idx(0)], area);
		atomicAdd(&mass[idx(1)], area);
		atomicAdd(&mass[idx(2)], area);
	}

	void StretchingConstraints::initialize(int n_faces, const MaterialParameters* material, const FaceIdx* indices, 
		const Vec3x* x, const Vec2x* uv, Scalar* mass)
	{
		m_num_faces = n_faces;

		cublasCreate(&m_handle);

		m_type = material->m_type;
		m_mu = material->m_thickness * material->m_youngs_modulus / (1 + material->m_possion_ratio) / 2.f;
		m_lambda = material->m_thickness * material->m_youngs_modulus * material->m_possion_ratio /
			(1 + material->m_possion_ratio) / (1 - 2 * material->m_possion_ratio);

		cudaMalloc((void**)&m_indices, n_faces * sizeof(FaceIdx));
		cudaMalloc((void**)&m_areas, n_faces * sizeof(Scalar));
		cudaMalloc((void**)&m_Dm, n_faces * sizeof(Mat2x));
		cudaMalloc((void**)&m_Dm_inv, n_faces * sizeof(Mat2x));
		cudaMalloc((void**)&m_energy, n_faces * sizeof(Scalar));

		cudaMemcpy(m_indices, indices, n_faces * sizeof(FaceIdx), cudaMemcpyHostToDevice);
		stretchingInitializeKernel <<< get_block_num(n_faces), g_block_dim >>> (
			n_faces, m_indices, x, uv, m_areas, m_Dm, m_Dm_inv, mass);
	}

	__global__ void computeStretchingEnergyStVKKernel(int n_faces, Scalar mu, Scalar lambda, const FaceIdx* indices, const Scalar* areas, 
		const Mat2x* Dm_inv, const Vec3x* x, Scalar* energies)
	{
		int fid = blockIdx.x * blockDim.x + threadIdx.x;
		if (fid >= n_faces) return;

		Vec3x p[3];
#pragma unroll
		for (int i = 0; i < 3; ++i) p[i] = x[indices[fid](i)];

		Mat3x2x F;
		F.setCol(0, p[0] - p[2]); F.setCol(1, p[1] - p[2]);
		F *= Dm_inv[fid];

		Mat2x E = 0.5f * (F.transpose() * F - Mat2x::Identity());

		energies[fid] = areas[fid] * (mu * E.squaredNorm() + 0.5f * lambda * E.trace() * E.trace());
	}

	Scalar StretchingConstraints::computeEnergy(const Vec3x* x)
	{
		Scalar total_energy;

		switch (m_type)
		{
		case MESH_TYPE_StVK:
			computeStretchingEnergyStVKKernel <<< get_block_num(m_num_faces), g_block_dim >>> (
				m_num_faces, m_mu, m_lambda, m_indices, m_areas, m_Dm_inv, x, m_energy);
			break;
		case MESH_TYPE_NEO_HOOKEAN:
			break;
		case MESH_TYPE_DATA_DRIVEN:
			break;
		case MESH_TYPE_COUNT:
			break;
		default:
			break;
		}

		CublasCaller<Scalar>::sum(m_handle, m_num_faces, m_energy, &total_energy);
		
		return total_energy;
	}

	__global__ void computeStretchingGradientAndHessianStVKKernel(int n_faces, Scalar mu, Scalar lambda, const FaceIdx* indices,
		const Scalar* areas, const Mat2x* Dm_invs, const Vec3x* x, Vec3x* grad, SparseMatrixWrapper hess)
	{
		int fid = blockIdx.x * blockDim.x + threadIdx.x;
		if (fid >= n_faces) return;

		FaceIdx idx = indices[fid];
		Vec3x p[3];
#pragma unroll
		for (int i = 0; i < 3; ++i) p[i] = x[idx(i)];

		Mat2x Dm_inv = Dm_invs[fid];
		Mat3x2x F;
		F.setCol(0, p[0] - p[2]); F.setCol(1, p[1] - p[2]);
		F *= Dm_inv;

		Mat2x E = 0.5f * (F.transpose() * F - Mat2x::Identity());

		// stress tensor P
		Mat3x2x P = F * (2 * mu * E + lambda * E.trace() * Mat2x::Identity());
		Mat2x Dm_invT = Dm_inv.transpose();
		P *= areas[fid] * Dm_invT;

		//P_test[fid] = P;

		atomicAddVec3x(&grad[idx(0)], P.col(0));
		atomicAddVec3x(&grad[idx(1)], P.col(1));
		atomicAddVec3x(&grad[idx(2)], -(P.col(0) + P.col(1)));

		// DPDF
		Tensor3232x DPDF;
		Mat3x2x deltaF; Mat2x deltaE;
		for (int i = 0; i < 3; ++i) for (int j = 0; j < 2; ++j)
		{
			deltaF.setZero();
			deltaF(i, j) = 1.f;
			deltaE = 0.5f * (deltaF.transpose() * F + F.transpose() * deltaF);
			DPDF(i, j) = 2.f * mu * (deltaF * E + F * deltaE) + lambda * (E.trace() * deltaF + deltaE.trace() * F);
		}
		DPDF.outerProd(areas[fid] * Dm_invT);
		DPDF.innerProd(Dm_invT);

		//!  00       01       -(00+01)
		//!  10       11       -(10+11)
		//!  -(00+10) -(01+11) 00+01+10+11
		Mat3x H_block[2][2];
		for (int k = 0; k < 2; ++k) for (int i = 0; i < 2; ++i)	// write top-left 2x2 block
		{
			for (int l = 0; l < 3; ++l) H_block[k][i].setRow(l, DPDF(l, k).col(i));
			hess.atomicAddBlock(idx(k), idx(i), H_block[k][i]);
		}
		hess.atomicAddBlock(idx(2), idx(0), -(H_block[0][0] + H_block[1][0]));
		hess.atomicAddBlock(idx(2), idx(1), -(H_block[0][1] + H_block[1][1]));
		hess.atomicAddBlock(idx(0), idx(2), -(H_block[0][0] + H_block[0][1]));
		hess.atomicAddBlock(idx(1), idx(2), -(H_block[1][0] + H_block[1][1]));
		hess.atomicAddBlock(idx(2), idx(2), H_block[0][0] + H_block[1][0] + H_block[0][1] + H_block[1][1]);
	}

	void StretchingConstraints::computeGradiantAndHessian(const Vec3x* x, Vec3x* grad, SparseMatrix& hess, bool definiteness_fix)
	{
		// FIXME: No definiteness fix yet !!!!!!!!!!!!!!!
		switch (m_type)
		{
		case MESH_TYPE_StVK:
			computeStretchingGradientAndHessianStVKKernel <<< get_block_num(m_num_faces), g_block_dim >>> (
				m_num_faces, m_mu, m_lambda, m_indices, m_areas, m_Dm_inv, x, grad, hess);
			break;
		case MESH_TYPE_NEO_HOOKEAN:
			break;
		case MESH_TYPE_DATA_DRIVEN:
			break;
		case MESH_TYPE_COUNT:
			break;
		default:
			break;
		}
	}

	/********************************* BENDING CONSTRAINTS ****************************/

	BendingConstraints::BendingConstraints():
		m_indices(NULL), m_stiffness(NULL), m_K(NULL), m_energy(NULL), m_handle(NULL), m_num_edges(0)
	{}

	BendingConstraints::~BendingConstraints()
	{
		if (m_indices) cudaFree(m_indices);
		if (m_stiffness) cudaFree(m_stiffness);
		if (m_K) cudaFree(m_K);
		if (m_energy) cudaFree(m_energy);

		if (m_handle) cublasDestroy(m_handle);
	}

	const EdgeIdx* BendingConstraints::getIndices() const
	{
		return m_indices;
	}

	__global__ void bendingInitializeKernel(int n_edges, Scalar scalar, const EdgeIdx* indices, const Vec3x* x, 
		Scalar* stiffness, Vec4x* Ks)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_edges) return;

		EdgeIdx idx = indices[i];
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

	void BendingConstraints::initialize(int n_edges, const MaterialParameters* material, const EdgeIdx* indices, const Vec3x* x)
	{
		m_num_edges = n_edges;

		cublasCreate(&m_handle);

		cudaMalloc((void**)&m_indices, n_edges * sizeof(EdgeIdx));
		cudaMalloc((void**)&m_stiffness, n_edges * sizeof(Scalar));
		cudaMalloc((void**)&m_K, n_edges * sizeof(Vec4x));
		cudaMalloc((void**)&m_energy, n_edges * sizeof(Scalar));

		cudaMemcpy(m_indices, indices, n_edges * sizeof(EdgeIdx), cudaMemcpyHostToDevice);
		bendingInitializeKernel <<< get_block_num(n_edges), g_block_dim >>> (
			n_edges, material->m_bending_stiffness, m_indices, x, m_stiffness, m_K);
	}

	__global__ void precomputeKernel(int n_edges, const EdgeIdx* indices, const Scalar* stiffness, const Vec4x* Ks, SparseMatrixWrapper A)
	{
		int eid = blockIdx.x * blockDim.x + threadIdx.x;
		if (eid >= n_edges) return;

		EdgeIdx idx = indices[eid];
		Vec4x K = Ks[eid];
		Scalar stiff = stiffness[eid];
#pragma unroll
		for (int i = 0; i < 4; ++i)
		{
#pragma unroll
			for (int j = 0; j < 4; ++j)
				A.atomicAddIdentity(idx(i), idx(j), stiff * K(i) * K(j));
		}
	}

	void BendingConstraints::precompute(SparseMatrix& sparse_mat)
	{
		precomputeKernel <<< get_block_num(m_num_edges), g_block_dim >>> (m_num_edges, m_indices, m_stiffness, m_K, sparse_mat);
	}

	__global__ void computeBendingEnergyKernel(int n_edges, const Scalar* stiffness, const EdgeIdx* indices, const Vec4x* Ks, 
		const Vec3x* x, Scalar* energies)
	{
		int eid = blockIdx.x * blockDim.x + threadIdx.x;
		if (eid >= n_edges) return;

		EdgeIdx idx = indices[eid];
		Vec3x p[4];
#pragma unroll
		for (int i = 0; i < 4; ++i) p[i] = x[idx(i)];

		Scalar energy = 0;
		Vec4x K = Ks[eid];
#pragma unroll
		for (int i = 0;i < 4;++i)
#pragma unroll
			for (int j = 0; j < 4; ++j)
			{
				energy += K(i) * K(j) * p[i].dot(p[j]);
			}

		energies[eid] = 0.5f * stiffness[eid] * energy;
	}

	Scalar BendingConstraints::computeEnergy(const Vec3x* x)
	{
		Scalar total_energy;
		computeBendingEnergyKernel <<< get_block_num(m_num_edges), g_block_dim >>> (
			m_num_edges, m_stiffness, m_indices, m_K, x, m_energy);
		CublasCaller<Scalar>::sum(m_handle, m_num_edges, m_energy, &total_energy);
		return total_energy;
	}

	__global__ void computeBendingGradientKernel(int n_edges, const Scalar* stiffness, const EdgeIdx* indices, 
		const Vec4x* Ks, const Vec3x* x, Vec3x* gradient)
	{
		int eid = blockIdx.x * blockDim.x + threadIdx.x;
		if (eid >= n_edges) return;

		EdgeIdx idx = indices[eid];
		Vec3x p[4];
#pragma unroll
		for (int i = 0; i < 4; ++i) p[i] = x[idx(i)];

		Vec4x K = Ks[eid];
		Vec3x g;
#pragma unroll
		for (int i = 0; i < 4; ++i)
		{
			g.setZero();
#pragma unroll
			for (int j = 0; j < 4; ++j)
			{
				g += K(i) * K(j) * p[j];
			}
			atomicAddVec3x(&gradient[idx(i)], stiffness[eid] * g);
		}
	}

	void BendingConstraints::computeGradiant(const Vec3x* x, Vec3x* gradient)
	{
		computeBendingGradientKernel <<< get_block_num(m_num_edges), g_block_dim >>> (m_num_edges, m_stiffness, m_indices, m_K, x, gradient);
	}

	/**************************** ATTACHMENT CONSTRAINTS *************************/

	AttachmentConstraints::AttachmentConstraints():
		m_indices(NULL), m_targets(NULL), m_energy(NULL), m_handle(NULL), m_num_fixed(0)
	{}

	AttachmentConstraints::~AttachmentConstraints()
	{
		if (m_indices) cudaFree(m_indices);
		if (m_targets) cudaFree(m_targets);
		if (m_energy) cudaFree(m_energy);

		if (m_handle) cublasDestroy(m_handle);
	}

	void AttachmentConstraints::initialize(int n_fixed, Scalar stiffness, const int* indices, const Vec3x* targets)
	{
		m_num_fixed = n_fixed;
		m_stiffness = stiffness;

		cublasCreate(&m_handle);

		if (n_fixed)
		{
			cudaMalloc((void**)&m_indices, n_fixed * sizeof(int));
			cudaMalloc((void**)&m_targets, n_fixed * sizeof(Vec3x));
			cudaMalloc((void**)&m_energy, n_fixed * sizeof(Scalar));

			cudaMemcpy(m_indices, indices, n_fixed * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(m_targets, targets, n_fixed * sizeof(Vec3x), cudaMemcpyHostToDevice);
		}
	}

	__global__ void computeAttachmentEnergyKernel(int n_attach, Scalar m_stiffness, const int* indices, const Vec3x* targets, const Vec3x* x, Scalar* energies)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_attach) return;

		energies[i] = 0.5 * m_stiffness * (targets[i] - x[indices[i]]).squareNorm();
	}

	Scalar AttachmentConstraints::computeEnergy(const Vec3x* x)
	{
		Scalar total_energy = 0;
		if (m_num_fixed)
		{
			computeAttachmentEnergyKernel <<< get_block_num(m_num_fixed), g_block_dim >>> (m_num_fixed, m_stiffness, m_indices, m_targets, x, m_energy);
			CublasCaller<Scalar>::sum(m_handle, m_num_fixed, m_energy, &total_energy);
		}
		return total_energy;
	}

	__global__ void computeAttachmentGradientAndHessianKernel(int n_attach, Scalar stiffness, const int* indices, const Vec3x* targets, const Vec3x* x, 
		Vec3x* grad, SparseMatrixWrapper hess)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_attach) return;

		int idx = indices[i];
		Vec3x g = stiffness * (x[idx] - targets[i]);
		atomicAddVec3x(&grad[idx], g);
		hess.atomicAddInIndentity(idx, stiffness);
	}

	void AttachmentConstraints::computeGradiantAndHessian(const Vec3x* x, Vec3x* gradient, SparseMatrix& hessian)
	{
		if (m_num_fixed)
		{
			computeAttachmentGradientAndHessianKernel << < get_block_num(m_num_fixed), g_block_dim >> > (
				m_num_fixed, m_stiffness, m_indices, m_targets, x, gradient, hessian);
		}
	}
}