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
#include <list>

#include "Utils/MathDef.h"
#include "Utils/json_loader.h"
#include "Utils/Cublas.h"
#include "Constraints.h"
#include "SparseMatrix.h"
#include "LinearSolver.h"

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
		std::vector<Vec3x> m_x;
		std::vector<Vec2x> m_uv;
		std::vector<FaceIdx> m_faces_idx;
		std::vector<EdgeIdx> m_edges_idx;

		//! Count
		int m_num_nodes;
		int m_num_faces;
		int m_num_edges;

		int m_param_idx;

		Cloth(Json::Value& json);
	};

	struct HandleGroup
	{
		std::vector<int> m_indices;
		std::vector<Vec3x> m_init_targets;
		std::vector<Vec3x> m_targets;

		int m_num_nodes;
		int m_cloth_idx;
		int m_motion_idx;

		HandleGroup(Json::Value& json, const std::vector<Cloth>& cloths);
		void update(Scalar time);
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

		//! Access device data (for rendering)
		const FaceIdx* getFaceIndices(int i) const;
		const Vec3x* getPositions() const;

		//! Access counting
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

		bool NewtonStep(Vec3x* v_next);
		
		void evaluateGradientAndHessian(const Vec3x* x);
		Scalar evaluateObjectiveValue(int i, const Vec3x* v_next);
		Scalar lineSearch(int i, const Vec3x* gradient_dir, const Vec3x* descent_dir);

		//! Simulation parameters
		Scalar m_time;
		Scalar m_duration;
		Scalar m_dt;

		IntegrationMethod m_integration_method;
		int m_integration_iterations;
		bool m_enable_line_search;
		Scalar m_ls_alpha;
		Scalar m_ls_beta;

		LinearSolverType m_linear_solver_type;
		int m_linear_solver_iterations;
		Scalar m_linear_solver_error;

		Vec3x m_gravity;
		bool m_enable_damping;
		Scalar m_damping_alpha;
		Scalar m_damping_beta;

		std::vector<MaterialParameters> m_materials;

		//! Scene
		std::vector<Cloth> m_cloths;
		std::vector<int> m_offsets;
		int m_num_cloths;
		int m_num_total_nodes;
		int m_num_total_faces;
		int m_num_total_egdes;
		int m_num_handles;

		//! Device pointors
		Vec3x* d_x;
		Vec3x* d_v;
		Vec2x* d_uv;
		Scalar* d_mass;

		std::vector<StretchingConstraints> m_stretching_constraints;
		std::vector<BendingConstraints> m_bending_constraints;
		std::vector<AttachmentConstraints> m_attachment_constraints;

		//! Animation
		std::vector<HandleGroup> m_handle_groups;

		//! Solver variables
		Vec3x* d_u;
		Vec3x* d_x_next;
		Vec3x* d_v_next;
		Vec3x* d_delta_v;
		Vec3x* d_g;
		Vec3x* d_Kv;
		Scalar* d_out;

		std::vector<SparseMatrix> m_stiffness_matrix;
		std::vector<SparseMatrix> m_A;
		std::vector<SparseMatrix> m_init_A;	// bending

		std::vector<LinearSolver> m_solvers;
		std::vector<CudaMvWrapper> m_mv;

		std::vector<Scalar> m_step_size;
		std::vector<bool> m_converged;

		//! CUDA handles
		cublasHandle_t m_cublas_handle;
	};
}

#endif // !CLOTH_SIM_H

