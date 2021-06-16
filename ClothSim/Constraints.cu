#include "Constraints.h"
#include <cuda_runtime.h>
#include "ClothSim.h"
#include "Utils/MathUtility.h"

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

	__global__ void stretchingInitializeKernel(int n_faces, const FaceIdx* indices, const FaceIdx* uv_indices, const Vec2x* uv, 
		Scalar* areas, Mat2x* Dms, Mat2x* Dm_invs, Scalar* mass)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_faces) return;

		FaceIdx idx = indices[i];
		FaceIdx uv_idx = uv_indices[i];
		Vec2x u0 = uv[uv_idx(0)], u1 = uv[uv_idx(1)], u2 = uv[uv_idx(2)];
		Vec2x e0 = u0 - u2, e1 = u1 - u2;

		Scalar area = 0.5f * fabs(e0(0) * e1(1) - e0(1) * e1(0));
		areas[i] = area;

		Mat2x Dm;
		Dm.setCol(0, e0);
		Dm.setCol(1, e1);
		Dms[i] = Dm;
		Dm_invs[i] = Dm.inverse();

		area /= 3;
		atomicAdd(&mass[idx(0)], area);
		atomicAdd(&mass[idx(1)], area);
		atomicAdd(&mass[idx(2)], area);
	}

	void StretchingConstraints::initialize(int n_faces, const MaterialParameters* material, const FaceIdx* indices, const FaceIdx* uv_indices,
		const Vec2x* uv, Scalar* mass)
	{
		m_num_faces = n_faces;

		cublasCreate(&m_handle);

		m_type = material->m_type;
		m_mu = material->m_thickness * material->m_youngs_modulus / (1 + material->m_possion_ratio) / 2.f;
		m_lambda = material->m_thickness * material->m_youngs_modulus * material->m_possion_ratio /
			(1 + material->m_possion_ratio) / (1 - 2 * material->m_possion_ratio);

		switch (m_type)
		{
		case cloth::MESH_TYPE_StVK:
			m_laplacian_coeff = 2 * m_mu + 1.0033 * m_lambda;
			break;
		case cloth::MESH_TYPE_NEO_HOOKEAN:
			m_laplacian_coeff = 2.0066 * m_mu + 1.0122 * m_lambda;
			break;
		case cloth::MESH_TYPE_DATA_DRIVEN:
			break;
		default:
			break;
		}

		cudaMalloc((void**)&m_indices, n_faces * sizeof(FaceIdx));
		cudaMalloc((void**)&m_areas, n_faces * sizeof(Scalar));
		cudaMalloc((void**)&m_Dm, n_faces * sizeof(Mat2x));
		cudaMalloc((void**)&m_Dm_inv, n_faces * sizeof(Mat2x));
		cudaMalloc((void**)&m_energy, n_faces * sizeof(Scalar));

		FaceIdx* d_uv_indices;
		cudaMalloc((void**)&d_uv_indices, n_faces * sizeof(FaceIdx));

		cudaMemcpy(m_indices, indices, n_faces * sizeof(FaceIdx), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uv_indices, uv_indices, n_faces * sizeof(FaceIdx), cudaMemcpyHostToDevice);
		stretchingInitializeKernel <<< get_block_num(n_faces), g_block_dim >>> (
			n_faces, m_indices, d_uv_indices, uv, m_areas, m_Dm, m_Dm_inv, mass);

		cudaFree(d_uv_indices);
	}

	__global__ void computeStretchingWeightedLaplacianKernel(int n_stretch, Scalar laplacian_coeff, const FaceIdx* indices, 
		const Scalar* areas, const Mat2x* Dm_invs, SparseMatrixWrapper L)
	{
		int fid = blockIdx.x * blockDim.x + threadIdx.x;
		if (fid >= n_stretch) return;

		FaceIdx idx = indices[fid];
		Mat2x Dm_inv = Dm_invs[fid];
		Mat2x local_L = areas[fid] * laplacian_coeff * Dm_inv * Dm_inv.transpose();

		for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
		{
			L.atomicAddIdentity(idx(i), idx(j), local_L(i, j));
		}

		for (int i = 0; i < 2; ++i)
		{
			Scalar sum = -local_L.row(i).sum();
			L.atomicAddIdentity(idx(2), idx(i), sum);
			L.atomicAddIdentity(idx(i), idx(2), sum);
		}

		L.atomicAddIdentity(idx(2), idx(2), local_L.sum());
	}

	void StretchingConstraints::computeWeightedLaplacian(SparseMatrix& Laplacian)
	{
		computeStretchingWeightedLaplacianKernel <<< get_block_num(m_num_faces), g_block_dim >>> (m_num_faces, m_laplacian_coeff, m_indices, m_areas, m_Dm_inv, Laplacian);
	}

	__global__ void computeStretchingEnergyStVKKernel(int n_faces, Scalar mu, Scalar lambda, const FaceIdx* indices, const Scalar* areas, 
		const Mat2x* Dm_inv, const Vec3x* x, Scalar* energies)
	{
		int fid = blockIdx.x * blockDim.x + threadIdx.x;
		if (fid >= n_faces) return;

		FaceIdx idx = indices[fid];
		Vec3x p0 = x[idx(0)], p1 = x[idx(1)], p2 = x[idx(2)];

		Mat3x2x F;
		F.setCol(0, p0 - p2); F.setCol(1, p1 - p2);
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

	__global__ void computeStretchingGradientStVKKernel(int n_faces, Scalar mu, Scalar lambda, const FaceIdx* indices,
		const Scalar* areas, const Mat2x* Dm_invs, const Vec3x* x, Vec3x* grad)
	{
		int fid = blockIdx.x * blockDim.x + threadIdx.x;
		if (fid >= n_faces) return;

		FaceIdx idx = indices[fid];
		Vec3x p0 = x[idx(0)], p1 = x[idx(1)], p2 = x[idx(2)];

		Mat3x2x F;
		Mat2x Dm_inv = Dm_invs[fid];
		F.setCol(0, p0 - p2); F.setCol(1, p1 - p2);
		F *= Dm_inv;

		Mat2x E = 0.5f * (F.transpose() * F - Mat2x::Identity());

		// stress tensor P
		Mat3x2x P = F * (2 * mu * E + lambda * E.trace() * Mat2x::Identity());
		Mat2x Dm_invT = Dm_inv.transpose();
		P *= areas[fid] * Dm_invT;

		atomicAddVec3x(&grad[idx(0)], P.col(0));
		atomicAddVec3x(&grad[idx(1)], P.col(1));
		atomicAddVec3x(&grad[idx(2)], -(P.col(0) + P.col(1)));
	}

	void StretchingConstraints::computeGradient(const Vec3x* x, Vec3x* grad)
	{
		switch (m_type)
		{
		case MESH_TYPE_StVK:
			computeStretchingGradientStVKKernel << < get_block_num(m_num_faces), g_block_dim >> > (
				m_num_faces, m_mu, m_lambda, m_indices, m_areas, m_Dm_inv, x, grad);
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

	__global__ void computeStretchingGradientAndHessianStVKKernel(int n_faces, Scalar mu, Scalar lambda, const FaceIdx* indices,
		const Scalar* areas, const Mat2x* Dm_invs, const Vec3x* x, Vec3x* grad, SparseMatrixWrapper hess, Vec9x* sigma_out)
	{
		int fid = blockIdx.x * blockDim.x + threadIdx.x;
		if (fid >= n_faces) return;

		extern __shared__ Mat9x shared_mem[];
		Mat9x& U = shared_mem[2 * threadIdx.x];
		Mat9x& V = shared_mem[2 * threadIdx.x + 1];

		int i, j, k, l;

		FaceIdx idx = indices[fid];
		Vec3x p0 = x[idx(0)], p1 = x[idx(1)], p2 = x[idx(2)];

		Mat3x2x F;
		Mat2x Dm_inv = Dm_invs[fid];
		F.setCol(0, p0 - p2); F.setCol(1, p1 - p2);
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
		for (i = 0; i < 3; ++i) for (j = 0; j < 2; ++j)
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
		for (k = 0; k < 2; ++k) for (i = 0; i < 2; ++i)	// write top-left 2x2 block
		{
			for (l = 0; l < 3; ++l) H_block[k][i].setRow(l, DPDF(l, k).col(i));
			U.setBlock(k, i, H_block[k][i]);
		}

		U.setBlock(2, 0, -(H_block[0][0] + H_block[1][0]));
		U.setBlock(2, 1, -(H_block[0][1] + H_block[1][1]));
		U.setBlock(0, 2, -(H_block[0][0] + H_block[0][1]));
		U.setBlock(1, 2, -(H_block[1][0] + H_block[1][1]));
		U.setBlock(2, 2, H_block[0][0] + H_block[1][0] + H_block[0][1] + H_block[1][1]);

		// definiteness fix
		Vec9x sigma;
		bool succ = SVDdecomp(U, V, sigma);
		sigma_out[fid] = sigma;
		if (succ)
		{
			//SVDreorder(*U, *V, sigma);

			Scalar smallest_sigma = 1e-6f;
			for (i = 0; i < 9; ++i) sigma(i) = fmax(sigma(i), smallest_sigma);

			Vec9x U_row;
			Scalar sum;
			for (i = 0; i < 9; ++i)
			{
				for (j = 0; j < 9; ++j)
					U_row(j) = U(i, j) * sigma(j);
				for (j = 0; j < 9; ++j)
				{
					sum = 0.0f;
					for (k = 0; k < 9; ++k)
						sum += U_row(k) * V(j, k);
					U(i, j) = sum;
				}
			}
		}

		for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
		{
			hess.atomicAddBlock(idx(i), idx(j), U.block<3, 3>(i, j));
		}
	}

	void StretchingConstraints::computeGradiantAndHessian(const Vec3x* x, Vec3x* grad, SparseMatrix& hess, bool definiteness_fix)
	{
		Vec9x* sigma;
		cudaMalloc((void**)&sigma, m_num_faces * sizeof(Vec9x));

		switch (m_type)
		{
		case MESH_TYPE_StVK:
			computeStretchingGradientAndHessianStVKKernel <<< get_block_num(m_num_faces), g_block_dim, g_block_dim * 2 * sizeof(Mat9x) >>> (
				m_num_faces, m_mu, m_lambda, m_indices, m_areas, m_Dm_inv, x, grad, hess, sigma);
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

		std::vector<Vec9x> test_S(m_num_faces);
		cudaMemcpy(test_S.data(), sigma, test_S.size() * sizeof(Vec9x), cudaMemcpyDeviceToHost);
		for (int i = 0; i < m_num_faces; ++i) {
			bool minus = false;
			for (int j = 0; j < 9; ++j)
				if (test_S[i](j) < 1e-6) minus = true;
			if (minus)
			{
				std::cout << "face " << i << ": "; 
				test_S[i].print();
				std::cout << '\n';
			}
		}
	}

	/********************************* BENDING CONSTRAINTS ****************************/

	BendingConstraints::BendingConstraints():
		m_indices(NULL), m_flat_stiffness(NULL), m_nonflat_stiffness(NULL), m_K(NULL), m_energy(NULL), m_handle(NULL), m_num_edges(0)
	{}

	BendingConstraints::~BendingConstraints()
	{
		if (m_indices) cudaFree(m_indices);
		if (m_flat_stiffness) cudaFree(m_flat_stiffness);
		if (m_nonflat_stiffness) cudaFree(m_nonflat_stiffness);
		if (m_K) cudaFree(m_K);
		if (m_energy) cudaFree(m_energy);

		if (m_handle) cublasDestroy(m_handle);
	}

	const EdgeIdx* BendingConstraints::getIndices() const
	{
		return m_indices;
	}

	__global__ void bendingInitializeKernel(int n_edges, Scalar stiffness, const EdgeIdx* indices, const Vec3x* x,
		Scalar* flat_stiffness, Scalar* nonflat_stiffness, Vec4x* Ks)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_edges) return;

		EdgeIdx idx = indices[i];

		Vec3x p0 = x[idx(0)], p1 = x[idx(1)], p2 = x[idx(2)], p3 = x[idx(3)];
		Vec3x e0 = p1 - p0, e1 = p2 - p0, e2 = p3 - p0;

		Scalar e0_norm2 = e0.squareNorm();
		Vec3x n0 = e0.cross(e1),  n1 = (e2).cross(e0);
		Scalar a0 = n0.norm(), a1 = n1.norm();
		Scalar e0_inv_h0 = e0_norm2 / a0, e0_inv_h1 = e0_norm2 / a1;

		Scalar s = e1.dot(e0) / e0_norm2, t = e2.dot(e0) / e0_norm2;

		Ks[i](0) = (1 - s) * e0_inv_h0 + (1 - t) * e0_inv_h1;
		Ks[i](1) = s * e0_inv_h0 + t * e0_inv_h1;
		Ks[i](2) = -e0_inv_h0;
		Ks[i](3) = -e0_inv_h1;

		Scalar cos_rest_angle = n0.normalized().dot(n1.normalized());
		flat_stiffness[i] = 6 * cos_rest_angle / (a0 + a1) * stiffness;
		Scalar triple_product = e0.cross(e1).dot(e2) * stiffness;
		nonflat_stiffness[i] = -6 * e0_norm2 * e0_norm2 / ((a0 + a1) * a0 * a0 * a1 * a1) * triple_product * stiffness;
	}

	void BendingConstraints::initialize(int n_edges, const MaterialParameters* material, const EdgeIdx* indices, const Vec3x* x)
	{
		m_num_edges = n_edges;

		cublasCreate(&m_handle);

		cudaMalloc((void**)&m_indices, n_edges * sizeof(EdgeIdx));
		cudaMalloc((void**)&m_flat_stiffness, n_edges * sizeof(Scalar));
		cudaMalloc((void**)&m_nonflat_stiffness, n_edges * sizeof(Scalar));
		cudaMalloc((void**)&m_K, n_edges * sizeof(Vec4x));
		cudaMalloc((void**)&m_energy, n_edges * sizeof(Scalar));

		cudaMemcpy(m_indices, indices, n_edges * sizeof(EdgeIdx), cudaMemcpyHostToDevice);
		bendingInitializeKernel <<< get_block_num(n_edges), g_block_dim >>> (
			n_edges, material->m_bending_stiffness, m_indices, x, m_flat_stiffness, m_nonflat_stiffness, m_K);
	}

	__global__ void precomputeKernel(int n_edges, const EdgeIdx* indices, const Scalar* flat_stiffness, const Vec4x* Ks, SparseMatrixWrapper A)
	{
		int eid = blockIdx.x * blockDim.x + threadIdx.x;
		if (eid >= n_edges) return;

		EdgeIdx idx = indices[eid];
		Vec4x K = Ks[eid];
		Scalar stiffness = flat_stiffness[eid];

		for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
		{
			A.atomicAddIdentity(idx(i), idx(j), stiffness * K(i) * K(j));
		}
	}

	void BendingConstraints::precompute(SparseMatrix& sparse_mat)
	{
		precomputeKernel <<< get_block_num(m_num_edges), g_block_dim >>> (m_num_edges, m_indices, m_flat_stiffness, m_K, sparse_mat);
	}

	__global__ void computeBendingEnergyKernel(int n_edges, const Scalar* flat_stiffness, const Scalar* nonflat_stiffness, const EdgeIdx* indices, const Vec4x* Ks, 
		const Vec3x* x, Scalar* energies)
	{
		int eid = blockIdx.x * blockDim.x + threadIdx.x;
		if (eid >= n_edges) return;

		EdgeIdx idx = indices[eid];
		Vec3x p[4];
		for (int i = 0; i < 4; ++i) p[i] = x[idx(i)];

		Scalar energy = 0;

		// flat term
		Vec4x K = Ks[eid];
		for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
		{
			energy += K(i) * K(j) * p[i].dot(p[j]);
		}

		// nonflat term
		Vec3x e0 = p[1] - p[0], e1 = p[2] - p[0], e2 = p[3] - p[0];

		energies[eid] = 0.5f * flat_stiffness[eid] * energy + nonflat_stiffness[eid] * e0.cross(e1).dot(e2);
	}

	Scalar BendingConstraints::computeEnergy(const Vec3x* x)
	{
		Scalar total_energy;
		computeBendingEnergyKernel <<< get_block_num(m_num_edges), g_block_dim >>> (
			m_num_edges, m_flat_stiffness, m_nonflat_stiffness, m_indices, m_K, x, m_energy);
		CublasCaller<Scalar>::sum(m_handle, m_num_edges, m_energy, &total_energy);
		return total_energy;
	}

	__global__ void computeBendingGradientKernel(int n_edges, const Scalar* flat_stiffness, const Scalar* nonflat_stiffness, const EdgeIdx* indices, 
		const Vec4x* Ks, const Vec3x* x, Vec3x* gradient)
	{
		int eid = blockIdx.x * blockDim.x + threadIdx.x;
		if (eid >= n_edges) return;

		EdgeIdx idx = indices[eid];
		Vec3x p[4];
		for (int i = 0; i < 4; ++i) p[i] = x[idx(i)];

		// flat term
		Vec4x K = Ks[eid];
		Vec3x g;
		for (int i = 0; i < 4; ++i)
		{
			g.setZero();
			for (int j = 0; j < 4; ++j)
			{
				g += K(i) * K(j) * p[j];
			}
			atomicAddVec3x(&gradient[idx(i)], flat_stiffness[eid] * g);
		}

		// nonflat term
		Vec3x e0 = p[1] - p[0], e1 = p[2] - p[0], e2 = p[3] - p[0];
		Scalar k_bar = nonflat_stiffness[eid];
		Vec3x g1 = k_bar * e1.cross(e2),
			  g2 = k_bar * e2.cross(e0),
			  g3 = k_bar * e0.cross(e1);

		atomicAddVec3x(&gradient[idx(0)], -(g1 + g2 + g3));
		atomicAddVec3x(&gradient[idx(1)], g1);
		atomicAddVec3x(&gradient[idx(2)], g2);
		atomicAddVec3x(&gradient[idx(3)], g3);
	}

	void BendingConstraints::computeGradiant(const Vec3x* x, Vec3x* gradient)
	{
		computeBendingGradientKernel <<< get_block_num(m_num_edges), g_block_dim >>> 
			(m_num_edges, m_flat_stiffness, m_nonflat_stiffness, m_indices, m_K, x, gradient);
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

	__global__ void computeAttachmentWeitedLaplacianKernel(int n_attach, Scalar stiffness, const int* indices, SparseMatrixWrapper L)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_attach) return;

		L.atomicAddIdentity(indices[i], stiffness);
	}

	void AttachmentConstraints::computeWeightedLaplacian(SparseMatrix& L)
	{
		computeAttachmentWeitedLaplacianKernel <<< get_block_num(m_num_fixed), g_block_dim >>> (m_num_fixed, m_stiffness, m_indices, L);
	}

	void AttachmentConstraints::update(int n_fixed, const int* indices, const Vec3x* targets)
	{
		m_num_fixed = n_fixed;
		if (n_fixed)
		{
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

	__global__ void computeAttachmentGradientKernel(int n_attach, Scalar stiffness, const int* indices, const Vec3x* targets, const Vec3x* x, Vec3x* grad)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_attach) return;

		int idx = indices[i];
		Vec3x g = stiffness * (x[idx] - targets[i]);
		atomicAddVec3x(&grad[idx], g);
	}

	void AttachmentConstraints::computeGradient(const Vec3x* x, Vec3x* gradient)
	{
		if (m_num_fixed)
		{
			computeAttachmentGradientKernel <<< get_block_num(m_num_fixed), g_block_dim >>> (
				m_num_fixed, m_stiffness, m_indices, m_targets, x, gradient);
		}
	}

	__global__ void computeAttachmentGradientAndHessianKernel(int n_attach, Scalar stiffness, const int* indices, const Vec3x* targets, const Vec3x* x, 
		Vec3x* grad, SparseMatrixWrapper hess)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_attach) return;

		int idx = indices[i];
		Vec3x g = stiffness * (x[idx] - targets[i]);
		atomicAddVec3x(&grad[idx], g);
		hess.atomicAddIdentity(idx, stiffness);
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