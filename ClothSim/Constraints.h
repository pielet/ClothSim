// Author: Shiyang Jia (jsy0325@foxmail.com)
// Data: 12/23/2020

#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include "Utils/MathDef.h"
#include "Utils/Cublas.h"
#include "SparseMatrix.h"

namespace cloth
{
	class MaterialParameters;
	enum MeshType;

	class StretchingConstraints
	{
	public:
		StretchingConstraints();
		~StretchingConstraints();

		void initialize(int n_faces, const MaterialParameters* material, const FaceIdx* indices, const Vec3x* x, const Vec2x* uv, Scalar* mass);
		
		Scalar computeEnergy(const Vec3x* x);
		void computeGradiantAndHessian(const Vec3x* x, Vec3x* gradient, SparseMatrix& hessian, bool definiteness_fix = true);

		const FaceIdx* getIndices() const;
	private:
		FaceIdx* m_indices;
		Scalar* m_areas;
		Mat2x* m_Dm;
		Mat2x* m_Dm_inv;		
		Scalar* m_energy;

		cublasHandle_t m_handle;

		Scalar m_mu;
		Scalar m_lambda;
		MeshType m_type;

		int m_num_faces;
	};

	class BendingConstraints
	{
	public:
		BendingConstraints();
		~BendingConstraints();

		void initialize(int n_edges, const MaterialParameters* material, const EdgeIdx* indices, const Vec3x* x);
		void precompute(SparseMatrix& sparse_mat);

		Scalar computeEnergy(const Vec3x* x);
		void computeGradiant(const Vec3x* x, Vec3x* gradient);

		const EdgeIdx* getIndices() const;

	private:
		EdgeIdx* m_indices;	// 4 * m_num_edges
		Scalar* m_stiffness;
		Vec4x* m_K;
		Scalar* m_energy;

		cublasHandle_t m_handle;

		int m_num_edges;
	};

	class AttachmentConstraints
	{
	public:
		AttachmentConstraints();
		~AttachmentConstraints();

		void initialize(int n_fixed, Scalar stiffness, const int* indices, const Vec3x* targets);
		
		void updateHandles(); 
		
		Scalar computeEnergy(const Vec3x* x);
		void computeGradiantAndHessian(const Vec3x* x, Vec3x* gradient, SparseMatrix& hessian);
	private:
		int* m_indices;
		Vec3x* m_targets;
		Scalar* m_energy;

		cublasHandle_t m_handle;

		Scalar m_stiffness;

		int m_num_fixed;
	};
}

#endif // !CONSTRAINTS_H
