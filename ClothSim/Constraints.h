// Author: Shiyang Jia (jsy0325@foxmail.com)
// Data: 12/23/2020

#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include "Utils/MathDef.h"
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

		void initialize(int n_faces, int offset, const MaterialParameters* material, const FaceIdx* indices, const Vec3x* x, const Vec2x* uv);
		void computeGradiant(Scalar* gradient);
		//void computeHessian(SparseMatrix* hessian, bool definiteness_fix = true);

		const FaceIdx* getIndices() const;
	private:
		FaceIdx* m_indices;
		Scalar* m_areas;
		Mat2x* m_Dm;
		Mat2x* m_Dm_inv;

		Scalar m_mu;
		Scalar m_lambda;
		MeshType m_type;

		int m_num_faces;
		int m_offset;
	};

	class BendingConstraints
	{
	public:
		BendingConstraints();
		~BendingConstraints();

		void intialize(int n_edges, int offset, const MaterialParameters* material, const EdgeIdx* indices, const Vec3x* x);
		void precompute(SparseMat3x& sparse_mat);

		const EdgeIdx* getIndices() const;

	private:
		EdgeIdx* m_indices;	// 4 * m_num_edges
		Scalar* m_stiffness;
		Vec4x* m_K;

		int m_num_edges;
		int m_offset;
	};
}

#endif // !CONSTRAINTS_H
