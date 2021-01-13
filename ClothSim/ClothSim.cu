#include "ClothSim.h"
#include <fstream>
#include <sstream>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

#include <iostream>

namespace cloth
{

	ClothSim::ClothSim(): 
		m_time(0), m_num_cloths(0), m_num_total_nodes(0), m_num_total_egdes(0), m_num_total_faces(0), m_num_handles(0),
		d_x(NULL), d_v(NULL), d_uv(NULL), d_mass(NULL), 
		d_u(NULL), d_g(NULL), d_x_next(NULL), d_v_next(NULL), d_delta_v(NULL), d_Kv(NULL), d_out(NULL)
	{
		if (sizeof(Scalar) == 8)
		{
			cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		}

		cublasCreate(&m_cublas_handle);
	}

	ClothSim::~ClothSim()
	{
		if (d_x) cudaFree(d_x);
		if (d_v) cudaFree(d_v);
		if (d_uv) cudaFree(d_uv);
		if (d_mass) cudaFree(d_mass);

		if (d_u) cudaFree(d_u);
		if (d_x_next) cudaFree(d_x_next);
		if (d_v_next) cudaFree(d_v_next);
		if (d_delta_v) cudaFree(d_delta_v);
		if (d_g) cudaFree(d_g);
		if (d_Kv) cudaFree(d_Kv);
		if (d_out) cudaFree(d_out);

		cublasDestroy(m_cublas_handle);
	}

	void ClothSim::initialize(std::string scene_fname)
	{
		// load json anf obj file
		loadScene(scene_fname);
		
		// allocates device memory
		int n_nodes = m_offsets[m_num_cloths];

		// initialize initial positions and constraints
		cudaMalloc((void**)&d_x, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_v, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_uv, n_nodes * sizeof(Vec2x));
		cudaMalloc((void**)&d_mass, n_nodes * sizeof(Scalar));

		cudaMalloc((void**)&d_u, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_g, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_x_next, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_v_next, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_delta_v, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_Kv, n_nodes * sizeof(Vec3x));
		cudaMalloc((void**)&d_out, n_nodes * sizeof(Scalar));

		cudaMemset(d_v, 0, n_nodes * sizeof(Vec3x));
		cudaMemset(d_mass, 0, n_nodes * sizeof(Scalar));

		m_stretching_constraints.resize(m_num_cloths);
		m_bending_constraints.resize(m_num_cloths);
		m_attachment_constraints.resize(m_num_cloths);
		m_stiffness_matrix.resize(m_num_cloths);
		m_A.resize(m_num_cloths);
		m_init_A.resize(m_num_cloths);
		m_solvers.resize(m_num_cloths);
		m_mv.resize(m_num_cloths);
		m_step_size.resize(m_num_cloths, 1.f);
		m_converged.resize(m_num_cloths, false);

		m_num_total_faces = 0;
		m_num_total_egdes = 0;
		m_num_total_nodes = m_offsets[m_num_cloths];

		int n_edges, n_faces, n_fixed;
		std::vector<int> handles;
		std::vector<Vec3x> targets;

		// FIXME: cuda stream (one for each cloth), 
		//        so we need setStream api for SparseMat and Constraints
		//        and set stream for cublas_handle
		//        and omp accumulation for m_num_total_xxx
		for (int i = 0; i < m_num_cloths; ++i)
		{
			n_nodes = m_cloths[i].m_num_nodes;
			n_faces = m_cloths[i].m_num_faces;
			n_edges = m_cloths[i].m_num_edges;

			m_num_total_faces += n_faces;
			m_num_total_egdes += n_edges;

			cudaMemcpy(d_x + m_offsets[i], m_cloths[i].m_x.data(), n_nodes * sizeof(Vec3x), cudaMemcpyHostToDevice);
			cudaMemcpy(d_uv + m_offsets[i], m_cloths[i].m_uv.data(), n_nodes * sizeof(Vec2x), cudaMemcpyHostToDevice);

			m_stretching_constraints[i].initialize(
				n_faces, &m_materials[m_cloths[i].m_param_idx], m_cloths[i].m_faces_idx.data(), &d_x[m_offsets[i]], &d_uv[m_offsets[i]], &d_mass[m_offsets[i]]);
			m_bending_constraints[i].initialize(
				n_edges, &m_materials[m_cloths[i].m_param_idx], m_cloths[i].m_edges_idx.data(), &d_x[m_offsets[i]]);

			m_A[i].initialize(n_nodes, n_faces, n_edges, m_stretching_constraints[i].getIndices(), m_bending_constraints[i].getIndices());
			m_init_A[i].initialize(m_A[i]);
			m_stiffness_matrix[i].initialize(m_A[i]);

			CublasCaller<Scalar>::scal(m_cublas_handle, n_nodes, &m_materials[m_cloths[i].m_param_idx].m_density, &d_mass[m_offsets[i]]);
			m_bending_constraints[i].precompute(m_init_A[i]);

			// attachment constraints
			handles.clear(); targets.clear(); n_fixed = 0;
			for (auto iter = m_handle_groups.begin(); iter != m_handle_groups.end(); ++iter)
			{
				if (iter->m_cloth_idx == i)
				{
					n_fixed += iter->m_num_nodes;
					handles.insert(handles.end(), iter->m_indices.begin(), iter->m_indices.end());
					targets.insert(targets.end(), iter->m_init_targets.begin(), iter->m_init_targets.end());
				}
			}
			m_attachment_constraints[i].initialize(n_fixed, m_materials[m_cloths[i].m_param_idx].m_attachment_stiffness, handles.data(), targets.data());
		
			m_solvers[i].initialize(&m_A[i], LinearSolver::Diagnal);
			m_mv[i].initialize(&m_stiffness_matrix[i], (Scalar*)&d_v_next[m_offsets[i]], (Scalar*)&d_Kv[m_offsets[i]]);
		}

		cudaDeviceSynchronize();
	}

	void ClothSim::loadScene(const std::string& scene_fname)
	{
		std::fstream fin(scene_fname);
		if (!fin.is_open())
		{
			throw std::runtime_error("[ClothSim::loadScene] Can NOT open json file: " + scene_fname);
		}

		Json::Value root;
		fin >> root;
		fin.close();

		// load parameters
		parse(m_duration, root["duration"], 1.0f);
		parse(m_dt, root["dt"], 0.001f);

		Json::Value value = root["nonlinear_solver"];
		parse(m_integration_method, value["type"], INTEGRATION_METHOD_NEWTON);
		parse(m_integration_iterations, value["iterations"], 5);
		parse(m_enable_line_search, value["enable_line_search"], true);
		parse(m_ls_alpha, value["ls_alpha"], 0.03f);
		parse(m_ls_beta, value["ls_beta"], 0.5f);

		value = root["linear_solver"];
		parse(m_linear_solver_type, value["type"], LINEAR_SOLVER_DIRECT_LLT);
		parse(m_linear_solver_iterations, value["iterations"], 200);
		parse(m_linear_solver_error, value["error"], 1e-3f);

		parse(m_gravity, root["gravity"]);
		parse(m_enable_damping, root["enable_damping"], true);
		parse(m_damping_alpha, root["damping_alpha"]);
		parse(m_damping_beta, root["damping_beta"]);

		m_materials.resize(root["parameters"].size());
		for (int i = 0; i < root["parameters"].size(); ++i)
		{
			MaterialParameters& param = m_materials[i];
			value = root["parameters"][i];
			parse(param.m_type, value["mesh_type"], MESH_TYPE_StVK);
			parse(param.m_youngs_modulus, value["youngs_modulus"], 2e12f);
			parse(param.m_possion_ratio, value["poisson_ratio"], 0.3f);
			parse(param.m_density, value["density"], 8.05f);
			parse(param.m_thickness, value["thickness"], 0.1f);
			parse(param.m_bending_stiffness, value["bending_stiffness"], 2e8f);
			parse(param.m_attachment_stiffness, value["attachment_stiffness"], 1e10f);
		}

		// load cloth models
		m_num_cloths = root["cloths"].size();
		m_offsets.resize(m_num_cloths + 1);
		m_offsets[0] = 0;
		for (int i = 0; i < m_num_cloths; ++i)
		{
			m_cloths.push_back(Cloth(root["cloths"][i]));
			m_offsets[i + 1] = m_offsets[i] + m_cloths.back().m_num_nodes;
		}

		// load handles and animations
		int n_handle_groups = root["handles"].size();
		int json_idx = 0;
		m_num_handles = 0;
		for (int i = 0; i < root["handles"].size(); ++i)
		{
			m_handle_groups.push_back(HandleGroup(root["handles"][i], m_cloths));
			m_num_handles += m_handle_groups.back().m_num_nodes;
		}
	}

	Cloth::Cloth(Json::Value& json)
	{
		std::string fname = json["mesh"].asString();
		Scalar rot_angle;
		Eigen::Vec3x rot_axis, trans;

		parse(rot_angle, json["transform"]["rotate_angle"], 0.f);
		parse(rot_axis, json["transform"]["rotate_axis"]);
		parse(trans, json["transform"]["translate"]);
		parse(m_param_idx, json["parameter"], 0);

		Eigen::Transformx transform = Eigen::Translationx(trans) * Eigen::AngleAxisx(degree_to_radian(rot_angle), rot_axis.normalized());

		std::fstream fin("../" + fname);
		if (!fin.is_open())
		{
			throw std::runtime_error("[ClothSim::loadScene] Can NOT open obj file: " + fname);
		}

		Eigen::Vec3x pos;
		Vec2x uv;
		FaceIdx face_idx;

		std::string line, token;
		while (std::getline(fin, line))
		{
			std::stringstream iss(line);
			iss >> token;
			if (token == "v")
			{
				iss >> pos(0) >> pos(1) >> pos(2);
				pos = transform.linear() * pos + transform.translation(); // FIXME: Eigen bug, I can't use transform * pos directly
				m_x.push_back(Vec3x(pos));
			}
			if (token == "vt")
			{
				iss >> uv(0) >> uv(1);
				m_uv.push_back(uv);
			}
			if (token == "f")
			{
				if (line.find('/') == std::string::npos)
					iss >> face_idx(0) >> face_idx(1) >> face_idx(2);
				else sscanf(line.c_str(), "f %d/%d %d/%d %d/%d", &face_idx(0), &face_idx(0), &face_idx(1), &face_idx(1), &face_idx(2), &face_idx(2));
				face_idx += -1;
				m_faces_idx.push_back(face_idx);
			}
		}
		m_num_nodes = m_x.size();
		m_num_faces = m_faces_idx.size();

		// generate edge list
		Eigen::SparseMatrix<int> edge_matrix(m_num_nodes, m_num_nodes);
		edge_matrix.setZero();
		
		int i0, i1, i2;
		for (int i = 0; i < m_num_faces; ++i)
		{
			i0 = m_faces_idx[i](0);
			i1 = m_faces_idx[i](1);
			i2 = m_faces_idx[i](2);
			if (edge_matrix.coeff(i0, i1) == 0)
			{
				m_edges_idx.push_back(EdgeIdx(i0, i1, i2, -1));
				edge_matrix.coeffRef(i0, i1) = m_edges_idx.size();
				edge_matrix.coeffRef(i1, i0) = m_edges_idx.size();
			}
			else
			{
				m_edges_idx[edge_matrix.coeff(i0, i1) - 1](3) = i2;
			}

			i0 = m_faces_idx[i](1);
			i1 = m_faces_idx[i](2);
			i2 = m_faces_idx[i](0);
			if (edge_matrix.coeff(i0, i1) == 0)
			{
				m_edges_idx.push_back(EdgeIdx(i0, i1, i2, -1));
				edge_matrix.coeffRef(i0, i1) = m_edges_idx.size();
				edge_matrix.coeffRef(i1, i0) = m_edges_idx.size();
			}
			else
			{
				m_edges_idx[edge_matrix.coeff(i0, i1) - 1](3) = i2;
			}

			i0 = m_faces_idx[i](0);
			i1 = m_faces_idx[i](2);
			i2 = m_faces_idx[i](1);
			if (edge_matrix.coeff(i0, i1) == 0)
			{
				m_edges_idx.push_back(EdgeIdx(i0, i1, i2, -1));
				edge_matrix.coeffRef(i0, i1) = m_edges_idx.size();
				edge_matrix.coeffRef(i1, i0) = m_edges_idx.size();
			}
			else
			{
				m_edges_idx[edge_matrix.coeff(i0, i1) - 1](3) = i2;
			}
		}

		m_edges_idx.erase(std::remove_if(
			m_edges_idx.begin(), m_edges_idx.end(), [](const EdgeIdx& idx) {
				return idx(3) == -1;
			}), m_edges_idx.end());
		m_num_edges = m_edges_idx.size();
	}

	HandleGroup::HandleGroup(Json::Value& json, const std::vector<Cloth>& cloths)
	{
		int start, end;
		parse(m_cloth_idx, json["cloth_idx"], 0);
		parse(start, json["start_idx"]);
		parse(end, json["end_idx"]);
		parse(m_motion_idx, json["motion"], -1);  // -1 means fixed

		for (int i = start; i < end; ++i)
		{
			m_indices.push_back(i);
			m_init_targets.push_back(cloths[m_cloth_idx].m_x[i]);
		}
		int node;
		for (int i = 0; i < json["nodes"].size(); ++i)
		{
			parse(node, json["nodes"][i]);
			m_indices.push_back(node);
			m_init_targets.push_back(cloths[m_cloth_idx].m_x[node]);
		}

		m_num_nodes = m_indices.size();
		m_targets.resize(m_num_nodes);
	}

	void HandleGroup::update(Scalar time)
	{
		if (m_motion_idx == -1)
		{
			m_targets = m_init_targets;
			return;
		}
	}

	__global__ void computeInitialGuessKernel(int n_nodes, Scalar h, const Vec3x* v, Vec3x gravity_accelaration, Vec3x* u)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_nodes) return;

		u[i] = v[i] + h * gravity_accelaration;
	}

	bool ClothSim::step()
	{
		m_time += m_dt;
		if (m_time > m_duration) return false;

		// Computes initial guess
		computeInitialGuessKernel <<< get_block_num(m_num_total_nodes), g_block_dim >>> (m_num_total_nodes, m_dt, d_v, m_gravity, d_u);
		CublasCaller<Scalar>::copy(m_cublas_handle, 3 * m_num_total_nodes, (Scalar*)d_u, (Scalar*)d_v);

		for (int i = 0; i < m_num_cloths; ++i)
		{
			m_step_size[i] = 1.f;
			m_converged[i] = false;
		}

		// Integration iteration
		std::cout << "[Start integration]\n";
		bool converged = false;
		for (int s = 0; s < m_integration_iterations; ++s)
		{
			std::cout << " Iteration " << s + 1 << ":  ";
			switch (m_integration_method)
			{
			case INTEGRATION_METHOD_NEWTON:
				converged = NewtonStep(d_v);
				break;
			case INTEGRATION_METHOD_LBFGS:
				break;
			default:
				break;
			}
			for (int i = 0; i < m_num_cloths; ++i)
			{
				if (!m_converged[i])
					CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * m_num_total_nodes, &m_step_size[i], (Scalar*)d_delta_v, (Scalar*)d_v);
			}
			if (converged) break;
		}

		// Updates states
		CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * m_num_total_nodes, &m_dt, (Scalar*)d_v, (Scalar*)d_x);

		// Update m_stiffness_matrix 
		if (m_enable_damping)
		{
			for (int i = 0; i < m_num_cloths; ++i)
			{
				m_stiffness_matrix[i].assign(m_init_A[i]); // bending
				m_attachment_constraints[i].computeGradiantAndHessian(&d_x[m_offsets[i]], &d_g[m_offsets[i]], m_stiffness_matrix[i]);
				m_stretching_constraints[i].computeGradiantAndHessian(&d_x[m_offsets[i]], &d_g[m_offsets[i]], m_stiffness_matrix[i]);
			}
		}

		return true;
	}

	bool ClothSim::NewtonStep(Vec3x* v_next)
	{
		CublasCaller<Scalar>::copy(m_cublas_handle, 3 * m_num_total_nodes, (Scalar*)d_x, (Scalar*)d_x_next);
		CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * m_num_total_nodes, &m_dt, (Scalar*)v_next, (Scalar*)d_x_next);

		//std::vector<Scalar> test;

		//test.resize(3 * getNumNodes(0));
		//cudaMemcpy(test.data(), v_next, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//std::cout << "v_t+1: ";
		//for (int i = 0; i < test.size(); ++i) {
		//	printf("%.10f ", test[i]);
		//}
		//std::cout << "\n\n";

		//test.resize(3 * getNumNodes(0));
		//cudaMemcpy(test.data(), d_x_next, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//std::cout << "x_t+1: ";
		//for (int i = 0; i < test.size(); ++i) {
		//	printf("%.10f ", test[i]);
		//}
		//std::cout << "\n\n";

		evaluateGradientAndHessian(d_x_next);

		std::vector<bool> status(m_num_cloths);
		for (int i = 0; i < m_num_cloths; ++i)
		{
			// Linear solver
			if (!m_converged[i])
				status[i] = m_solvers[i].cholesky((Scalar*)&d_g[m_offsets[i]], (Scalar*)&d_delta_v[m_offsets[i]]);
		}

		Scalar minus_one = -1;
		CublasCaller<Scalar>::scal(m_cublas_handle, 3 * m_num_total_nodes, &minus_one, (Scalar*)d_delta_v);

		//test.resize(3 * getNumNodes(0));
		//cudaMemcpy(test.data(), d_delta_v, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//std::cout << "delta v: ";
		//for (int i = 0; i < test.size(); ++i) {
		//	std::cout << test[i] << ' ';
		//}
		//std::cout << "\n\n";

		std::vector<Scalar> obj_values(m_num_cloths, 0.f);
		for (int i = 0; i < m_num_cloths; ++i)
		{
			if (!m_converged[i]) obj_values[i] = lineSearch(i, d_g, d_delta_v);
			if (m_step_size[i] < 1e-5f) m_converged[i] = true;
			std::cout << "objective value " << obj_values[i] << "  step size " << m_step_size[i] << std::endl;
		}

		bool all_pass = true;
		for (int i = 0; i < m_num_cloths; ++i)
			all_pass &= m_converged[i];

		return all_pass;
	}

	__global__ void computeGradientWithDampingKernel(int n, Scalar h, Scalar alpha, Scalar beta, Scalar* M, const Vec3x* v_next, const Vec3x* u, const Vec3x* Kv,
		const Vec3x* grad_E, Vec3x* grad_f)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		grad_f[i] = (1 + h * alpha) * M[i] * v_next[i] - M[i] * u[i] + h * beta * Kv[i] + h * grad_E[i];
	}

	__global__ void computeGradientWithoutDampingKernel(int n, Scalar h, const Scalar* M, const Vec3x* v_next, const Vec3x* u, const Vec3x* grad_E, Vec3x* grad_f)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		grad_f[i] = M[i] * (v_next[i] - u[i]) + h * grad_E[i];
	}

	void ClothSim::evaluateGradientAndHessian(const Vec3x* x)
	{
		cudaMemset(d_g, 0, m_num_total_nodes * sizeof(Vec3x));

		// Evaluates gradient and hessian of inner energy
		for (int i = 0; i < m_num_cloths; ++i)
		{
			if (!m_converged[i])
			{
				//m_A[i].setZero();
				m_A[i].assign(m_init_A[i]); // bending hessian
				m_bending_constraints[i].computeGradiant(&x[m_offsets[i]], &d_g[m_offsets[i]]); // bending gradient
				m_stretching_constraints[i].computeGradiantAndHessian(&x[m_offsets[i]], &d_g[m_offsets[i]], m_A[i], true);
				m_attachment_constraints[i].computeGradiantAndHessian(&x[m_offsets[i]], &d_g[m_offsets[i]], m_A[i]);

				if (m_enable_damping) m_mv[i].mv(); // K * v_t+1
			}
		}

		//std::vector<Scalar> test;

		//test.resize(3 * getNumNodes(0));
		//cudaMemcpy(test.data(), d_g, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < test.size(); ++i) {
		//	std::cout << test[i] << ' ';
		//}
		//std::cout << "\n\n";

		//test.resize(m_A[0].getnnz());
		//cudaMemcpy(test.data(), m_A[0].getValue(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < 12; ++i)
		//{
		//	for (int j = 0; j < 12; ++j)
		//		std::cout << test[12 * i + j] << ' ';
		//	std::cout << "\n";
		//}
		//std::cout << "\n";

		// \grad f = M * (x_t+1 - y) + h * (\alpha M + \beta K) * (x_t+1 - x_t) + h^2 * \grad E(x_t+1)
		// \grad f = M * (v_t+1 - u) + h * (\alpha M + \beta K) * v_t+1 + h * \grad E(x_t + h * v_t+1)
		if (m_enable_damping) {
			computeGradientWithDampingKernel <<< get_block_num(m_num_total_nodes), g_block_dim >>> (
				m_num_total_nodes, m_dt, m_damping_alpha, m_damping_beta, d_mass, d_v_next, d_u, d_Kv, d_g, d_g);
		}
		else {
			computeGradientWithoutDampingKernel <<< get_block_num(m_num_total_nodes), g_block_dim >>> (
				m_num_total_nodes, m_dt, d_mass, d_v_next, d_u, d_g, d_g);
		}

		//test.resize(3 * getNumNodes(0));
		//cudaMemcpy(test.data(), d_g, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < test.size(); ++i) {
		//	std::cout << test[i] << ' ';
		//}
		//std::cout << "\n\n";

		// \grad^2 f = M + h^2 * \grad^2 E
		// \grad^2 f = M + h^2 * \grad^2 E
		// \grad^2 f = M + h^2 * \grad^2 E + h * (\alpha M + \beta K) = (1 + h * \alpha) M + (h * \beta) K + (h*h) \grad^2 E
		Scalar h2 = m_dt * m_dt, h_beta = m_dt * m_damping_beta;
		for (int i = 0; i < m_num_cloths; ++i)
		{
			if (!m_converged[i])
			{
				if (m_enable_damping)
				{
					CublasCaller<Scalar>::scal(m_cublas_handle, m_A[i].getnnz(), &h2, m_A[i].getValue());
					CublasCaller<Scalar>::axpy(m_cublas_handle, m_A[i].getnnz(), &h_beta, m_stiffness_matrix[i].getValue(), m_A[i].getValue());
					m_A[i].addInDiagonal(&d_mass[m_offsets[i]], 1 + m_dt * m_damping_alpha);
				}
				else
				{
					CublasCaller<Scalar>::scal(m_cublas_handle, m_A[i].getnnz(), &h2, m_A[i].getValue());
					m_A[i].addInDiagonal(&d_mass[m_offsets[i]]);
				}
			}
		}

		//test.resize(m_A[0].getnnz());
		//cudaMemcpy(test.data(), m_A[0].getValue(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < 12; ++i)
		//{
		//	for (int j = 0; j < 12; ++j)
		//		std::cout << test[12 * i + j] << ' ';
		//	std::cout << "\n";
		//}
		//std::cout << "\n";
	}

	__global__ void computeInertiaWithDampingKernel(int n, Scalar h, Scalar alpha, Scalar beta, const Scalar* M, const Vec3x* v_next, const Vec3x* u, 
		const Vec3x* Kv, Scalar* out)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		out[i] = 0.5 * M[i] * (v_next[i] - u[i]).squareNorm() + 0.5 * h * (alpha * M[i] * v_next[i].squareNorm() + beta * v_next[i].dot(Kv[i]));
	}

	__global__ void computeInertiaWithoutDampingKernel(int n, const Scalar* M, const Vec3x* v_next, const Vec3x* u, Scalar* out)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		out[i] = 0.5f * M[i] * (v_next[i] - u[i]).squareNorm();
	}

	Scalar ClothSim::evaluateObjectiveValue(int i, const Vec3x* v_next)
	{
		int n_nodes = getNumNodes(i);

		CublasCaller<Scalar>::copy(m_cublas_handle, 3 * n_nodes, (Scalar*)&d_x[m_offsets[i]], (Scalar*)&d_x_next[m_offsets[i]]);
		CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * n_nodes, &m_dt, (Scalar*)v_next, (Scalar*)&d_x_next[m_offsets[i]]);

		// f = 1/2 * ||v_t+1 - u||_M + E + 
		//     1/2 * h * ||v_t+1||_(\alpha M + \beta K)
		Scalar inertia_energy;

		if (m_enable_damping)
		{
			m_mv[i].mv(); // FIXME: assuming v_next is d_v_next[m_offsets[i]]
			computeInertiaWithDampingKernel <<< get_block_num(n_nodes), g_block_dim >>> (
				n_nodes, m_dt, m_damping_alpha, m_damping_beta, &d_mass[m_offsets[i]], v_next, &d_u[m_offsets[i]], &d_Kv[m_offsets[i]], &d_out[m_offsets[i]]);
			CublasCaller<Scalar>::sum(m_cublas_handle, n_nodes, &d_out[m_offsets[i]], &inertia_energy);
		}
		else
		{
			computeInertiaWithoutDampingKernel <<< get_block_num(n_nodes), g_block_dim >>> (
				n_nodes, &d_mass[m_offsets[i]], v_next, &d_u[m_offsets[i]], &d_out[m_offsets[i]]);
			CublasCaller<Scalar>::sum(m_cublas_handle, n_nodes, &d_out[m_offsets[i]], &inertia_energy);
		}

		Scalar inner_energy = 0;
		inner_energy += m_attachment_constraints[i].computeEnergy(&d_x_next[m_offsets[i]]);
		inner_energy += m_bending_constraints[i].computeEnergy(&d_x_next[m_offsets[i]]);
		inner_energy += m_stretching_constraints[i].computeEnergy(&d_x_next[m_offsets[i]]);

		return inertia_energy + inner_energy;
	}

	Scalar ClothSim::lineSearch(int i, const Vec3x* gradient_dir, const Vec3x* descent_dir)
	{
		int n_nodes = getNumNodes(i);
		Scalar current_obj_value = evaluateObjectiveValue(i, &d_v[m_offsets[i]]);

		if (m_enable_line_search)
		{
			Scalar next_obj_value, rhs;
			Scalar grad_dot_desc;
			CublasCaller<Scalar>::dot(m_cublas_handle, 3 * n_nodes, (Scalar*)gradient_dir, (Scalar*)descent_dir, &grad_dot_desc);

			Scalar& step_size = m_step_size[i];
			step_size = std::min(1.f, 2 * std::max(1e-5f, step_size)) / m_ls_beta;

			do {
				step_size *= m_ls_beta;
				if (step_size < 1e-5) break;

				CublasCaller<Scalar>::copy(m_cublas_handle, 3 * n_nodes, (Scalar*)&d_v[m_offsets[i]], (Scalar*)&d_v_next[m_offsets[i]]);
				CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * n_nodes, &step_size, (Scalar*)descent_dir, (Scalar*)&d_v_next[m_offsets[i]]);

				next_obj_value = evaluateObjectiveValue(i, &d_v_next[m_offsets[i]]);

				rhs = current_obj_value + m_ls_alpha * step_size * grad_dot_desc;

			} while (next_obj_value > rhs);

			if (step_size < 1e-5)
			{
				step_size = 0;
				return current_obj_value;
			}
			else return next_obj_value;
		}
		else return current_obj_value;
	}

	const FaceIdx* ClothSim::getFaceIndices(int i) const
	{
		return m_stretching_constraints[i].getIndices();
	}

	const Vec3x* ClothSim::getPositions() const
	{
		return d_x;
	}

	int ClothSim::getOffset(int i) const
	{
		return m_offsets[i];
	}

	int ClothSim::getNumCloths() const
	{
		return m_num_cloths;
	}

	int ClothSim::getNumNodes(int i) const
	{
		return m_cloths[i].m_num_nodes;
	}

	int ClothSim::getNumFaces(int i) const
	{
		return m_cloths[i].m_num_faces;
	}

	int ClothSim::getNumEdges(int i) const
	{
		return m_cloths[i].m_num_edges;
	}

	int ClothSim::getNumTotalNodes() const
	{
		return m_num_total_nodes;
	}

	int ClothSim::getNumTotalFaces() const
	{
		return m_num_total_faces;
	}

	int ClothSim::getNumTotalEdges() const
	{
		return m_num_total_egdes;
	}
}