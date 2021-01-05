// Author: Shiyang Jia (jsy0325@foxmail.com)
// Data: 12/23/2020

#ifndef CLOTH_SIM_H
#define CLOTH_SIM_H

//#define MAKEDLL
//
//#ifdef MAKEDLL
//#  define EXPORT __declspec(dllexport)
//#else
//#  define EXPORT __declspec(dllimport)
//#endif

#include <string>
#include <vector>

#include "Utils/MathDef.h"
#include "Utils/json_loader.h"
#include "Constraints.h"
#include "SparseMatrix.h"

namespace cloth
{
	enum MeshType
	{
		MESH_TYPE_StVK,
		MESH_TYPE_NEO_HOOKEAN,
		MESH_TYPE_DATA_DRIVEN,

		MESH_TYPE_COUNT
	};

	enum IntegrationMethod
	{
		INTEGRATION_METHOD_NEWTON,
		INTEGRATION_METHOD_LBFGS,

		INTEGRATION_METHOD_COUNT
	};

	enum LinearSolverType
	{
		LINEAR_SOLVER_DIRECT_LLT,
		LINEAR_SOLVER_CONJ_GRAD,

		LINEAR_SOLVER_COUNT
	};

	struct MaterialParameters
	{
		MeshType m_type;
		Scalar m_youngs_modulus;
		Scalar m_possion_ratio;
		Scalar m_density;
		Scalar m_thickness;
		Scalar m_bending_stiffness;
		Scalar m_attachment_stiffness;
	};

	struct Cloth
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		//! Input models
		Eigen::VecXx m_x;	// 3 * num_nodes
		Eigen::VecXx m_uv;	// 2 * num_nodes
		std::vector<FaceIdx> m_faces_idx;
		std::vector<EdgeIdx> m_edges_idx;

		//! Count
		int m_num_nodes;
		int m_num_faces;
		int m_num_edges;

		int m_param_idx;

		int initialize(Json::Value& json);
	};

	class /*EXPORT*/ ClothSim
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		ClothSim();
		virtual ~ClothSim();

		//! Reads config file (.json) and objective model (.obj), allocates memory on gpu and initializes constraints
		void initialize(std::string filename);

		//! Step once, returns false if the simulation finished.
		bool step();

		//! Accesses device data (for rendering)
		const FaceIdx* getFaceIndices(int i) const;
		const Vec3x* getPositions() const;

		//! Accesses counting
		int getOffset(int i) const;
		int getNumCloths() const;
		int getNumNodes(int i) const;
		int getNumFaces(int i) const;
		int getNumEdges(int i) const;
		int getNumTotalNodes() const;
		int getNumTotalFaces() const;
		int getNumTotalEdges() const;

	private:
		void loadScene(const std::string& fname);
		void loadObj(Cloth& cloth_obj, Json::Value& json);

		//! Simulation parameters
		Scalar m_time;
		Scalar m_dt;

		IntegrationMethod m_integration_method;
		int m_integration_iterations;
		Scalar m_ls_alpha;
		Scalar m_ls_beta;

		LinearSolverType m_linear_solver_type;
		int m_linear_solver_iterations;
		Scalar m_linear_solver_error;

		Eigen::Vec3x m_gravity;
		Scalar m_damping_coefficient;

		std::vector<MaterialParameters> m_materials;

		//! Scene
		std::vector<Cloth> m_cloths;
		std::vector<int> m_offsets;
		int m_num_cloths;
		int m_num_total_nodes;
		int m_num_total_faces;
		int m_num_total_egdes;

		//! Device pointors
		Vec3x* d_x;
		Vec3x* d_v;
		Vec2x* d_uv;

		Scalar* d_mass;

		std::vector<StretchingConstraints> m_stretching_constraints;
		std::vector<BendingConstraints> m_bending_constraints;

		std::vector<SparseMat3x> m_stiffness_matrix;
		std::vector<SparseMat3x> m_A;
		std::vector<SparseMat3x> m_init_A;	// bending
	};
}

#endif // !CLOTH_SIM_H

