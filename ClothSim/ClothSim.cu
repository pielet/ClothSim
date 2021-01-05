#include "ClothSim.h"
#include <fstream>
#include <sstream>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

namespace cloth
{

	ClothSim::ClothSim(): 
		m_num_cloths(0), m_num_total_egdes(0), m_num_total_faces(0),
		d_x(NULL), d_v(NULL), d_uv(NULL)
	{
		if (sizeof(Scalar) == 8)
		{
			cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		}
	}

	ClothSim::~ClothSim()
	{
		if (d_x) cudaFree(d_x);
		if (d_v) cudaFree(d_v);
		if (d_uv) cudaFree(d_uv);
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
		cudaMemset(d_v, 0, n_nodes * sizeof(Vec3x));

		m_stretching_constraints.resize(m_num_cloths);
		m_bending_constraints.resize(m_num_cloths);
		m_stiffness_matrix.resize(m_num_cloths);
		m_A.resize(m_num_cloths);
		m_init_A.resize(m_num_cloths);

		m_num_total_faces = 0;
		m_num_total_egdes = 0;
		m_num_total_nodes = m_offsets[m_num_cloths];

		int n_edges, n_faces;

		for (int i = 0; i < m_num_cloths; ++i)
		{
			n_nodes = m_cloths[i].m_num_nodes;
			n_faces = m_cloths[i].m_num_faces;
			n_edges = m_cloths[i].m_num_edges;

			m_num_total_faces += n_faces;
			m_num_total_egdes += n_edges;

			cudaMemcpy(d_x + m_offsets[i], m_cloths[i].m_x.data(), n_nodes * sizeof(Vec3x), cudaMemcpyHostToDevice);
			cudaMemcpy(d_uv + m_offsets[i], m_cloths[i].m_uv.data(), n_nodes * sizeof(Vec2x), cudaMemcpyHostToDevice);

			m_stretching_constraints[i].initialize(n_faces, m_offsets[i], &m_materials[m_cloths[i].m_param_idx], m_cloths[i].m_faces_idx.data(), d_x, d_uv);
			m_bending_constraints[i].intialize(n_edges, m_offsets[i], &m_materials[m_cloths[i].m_param_idx], m_cloths[i].m_edges_idx.data(), d_x);

			m_A[i].initialize(n_nodes, n_faces, n_edges, m_stretching_constraints[i].getIndices(), m_bending_constraints[i].getIndices());
			m_init_A[i].initialize(m_A[i]);
			m_stiffness_matrix[i].initialize(m_A[i]);
		}
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
		parse(m_time, root["time"], 1.0f);
		parse(m_dt, root["dt"], 0.001f);

		Json::Value value = root["nonlinear_solver"];
		parse(m_integration_method, value["type"], INTEGRATION_METHOD_NEWTON);
		parse(m_integration_iterations, value["iterations"], 5);
		parse(m_ls_alpha, value["ls_alpha"], 0.03f);
		parse(m_ls_beta, value["ls_beta"], 0.5f);

		value = root["linear_solver"];
		parse(m_linear_solver_type, value["type"], LINEAR_SOLVER_DIRECT_LLT);
		parse(m_linear_solver_iterations, value["iterations"], 200);
		parse(m_linear_solver_error, value["error"], 1e-3f);

		parse(m_gravity, root["gravity"]);
		parse(m_damping_coefficient, root["damping"]);

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
		m_cloths.resize(m_num_cloths);
		m_offsets.resize(m_num_cloths + 1);
		m_offsets[0] = 0;
		for (int i = 0; i < m_num_cloths; ++i)
		{
			m_offsets[i + 1] = m_offsets[i] + m_cloths[i].initialize(root["cloths"][i]);
		}
	}

	int Cloth::initialize(Json::Value& json)
	{
		std::string fname = json["mesh"].asString();
		Scalar rot_angle;
		Eigen::Vec3x rot_axis, trans;

		parse(rot_angle, json["transform"]["rotate_angle"], 0.f);
		parse(rot_axis, json["transform"]["rotate_axis"]);
		parse(trans, json["transform"]["translate"]);
		parse(m_param_idx, json["parameter"], 0);

		Eigen::Transformx transform = Eigen::Translationx(trans) * Eigen::AngleAxisx(rot_angle, rot_axis.normalized());

		std::fstream fin("../" + fname);
		if (!fin.is_open())
		{
			throw std::runtime_error("[ClothSim::loadScene] Can NOT open obj file: " + fname);
		}

		Eigen::Vec3x pos;
		Eigen::Vec2x uv;
		FaceIdx face_idx;
		std::vector<Eigen::Vec3x, Eigen::aligned_allocator<Eigen::Vec3x> > positions;
		std::vector<Eigen::Vec2x, Eigen::aligned_allocator<Eigen::Vec2x> > uvs;

		std::string line, token;
		while (std::getline(fin, line))
		{
			std::stringstream iss(line);
			iss >> token;
			if (token == "v")
			{
				iss >> pos(0) >> pos(1) >> pos(2);
				pos = transform.linear() * pos + transform.translation(); // FIXME: Eigen's bug, I can't use transform * pos
				positions.push_back(pos);
			}
			if (token == "vt")
			{
				iss >> uv(0) >> uv(1);
				uvs.push_back(uv);
			}
			if (token == "f")
			{
				iss >> face_idx(0) >> face_idx(1) >> face_idx(2);
				face_idx += -1;
				m_faces_idx.push_back(face_idx);
			}
		}

		// compact vectors
		m_num_nodes = positions.size();
		m_num_faces = m_faces_idx.size();
		m_x.resize(3 * m_num_nodes);
		m_uv.resize(2 * m_num_nodes);
		for (int i = 0; i < m_num_nodes; ++i)
		{
			m_x.segment<3>(3 * i) = positions[i];
			m_uv.segment<2>(2 * i) = uvs[i];
		}

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

		return m_num_nodes;
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