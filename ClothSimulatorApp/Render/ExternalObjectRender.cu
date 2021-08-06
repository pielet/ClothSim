#include "ExternalObjectRender.h"
#include <cmath>
#include <cuda_gl_interop.h>
#include "Shader.h"

namespace cloth
{
	template <typename T1, typename T2, int n>
	Vec<T1, n> cast(const Vec<T2, n>& vec)
	{
		Vec<T1, n> new_vec;
		for (int i = 0; i < n; ++i)
			new_vec(i) = T1(vec(i));
		return new_vec;
	}

	template <typename T1, typename T2, int n>
	Eigen::Matrix<T1, n, 1> cast_2_Eigen(const Vec<T2, n>& vec)
	{
		Eigen::Matrix<T1, n, 1> eigen_vec;
		for (int i = 0; i < n; ++i)
			eigen_vec(i) = T1(vec(i));
		return eigen_vec;
	}

	ExternalObjectRender::ExternalObjectRender(const ClothSim* cloth_sim)
	{
		m_dt = cloth_sim->getDt();
		m_face_color = Eigen::Vec3f(128.f, 128.f, 128.f) / 255.f;
		m_frame_color = Eigen::Vec3f(96.f, 96.f, 96.f) / 255.f;
	}

	SphereRender::SphereRender(const ClothSim* cloth_sim, const Sphere* sphere) :
		ExternalObjectRender(cloth_sim), m_sphere(sphere)
	{
		m_origin = cast<float>(sphere->m_origin);

		// Construct vertices and indices
		int n_row = 10, n_col = 20;
		m_num_node = (n_row - 1) * n_col + 2;
		m_num_tri = 2 * n_col;
		m_num_quad = (n_row - 2) * n_col;

		std::vector<Vec3f> vertices;
		std::vector<unsigned> indices;

		// fill vertices
		float alpha = M_PI / n_row;
		float beta = 2 * M_PI / n_col;
		for (int i = 1; i < n_row; ++i) for (int j = 0; j < n_col; ++j)
		{
			vertices.emplace_back(sin(i * alpha) * cos(j * beta), cos(i * alpha), sin(i * alpha) * sin(j * beta));
		}
		vertices.emplace_back(0.f, 1.f, 0.f);
		vertices.emplace_back(0.f, -1.f, 0.f);

		assert(vertices.size() == m_num_node);

		// fill faces
		for (int i = 0; i < n_row - 2; ++i) for (int j = 0; j < n_col; ++j)
		{
			indices.push_back(i * n_col + j);
			indices.push_back(i * n_col + (j + 1) % n_col);
			indices.push_back((i + 1) * n_col + (j + 1) % n_col);
			indices.push_back((i + 1) * n_col + j);
		}

		int upper_idx = m_num_node - 2, lower_idx = m_num_node - 1;
		int last_row_idx = (n_row - 2) * n_col;
		for (int i = 0; i < n_col; ++i)
		{
			indices.push_back(upper_idx);
			indices.push_back(i);
			indices.push_back((i + 1) % n_col);

			indices.push_back(lower_idx);
			indices.push_back(last_row_idx + i);
			indices.push_back(last_row_idx + (i + 1) % n_col);
		}

		assert(indices.size() == 3 * m_num_tri + 4 * m_num_quad);

		// scale and translate
		for (auto& v : vertices) v = sphere->m_radius * v + m_origin;

		// Buffer data
		glGenBuffers(1, &m_verticesBuffer);
		glGenBuffers(1, &m_indicesBuffer);

		glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vec3f), vertices.data(), GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);

		// Register vertice buffer
		cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_verticesBuffer, cudaGraphicsMapFlagsWriteDiscard);
	}

	SphereRender::~SphereRender()
	{
		cudaGraphicsUnregisterResource(m_cudaResource);

		glDeleteBuffers(1, &m_verticesBuffer);
		glDeleteBuffers(1, &m_indicesBuffer);
	}

	__global__ void updateSphereVerticesKernel(int n, float dt, Vec3f origin, Vec3f linear_vel, Vec3f angular_vel, Vec3f* verts)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n) return;

		Vec3f& vert = verts[i];

		float theta = angular_vel.norm() * dt;
		if (fabs(theta) > EPS)
		{
			Vec3f z = angular_vel.normalized();
			Vec3f r = vert - origin;
			vert = origin + r.dot(z) * z + sinf(theta) * z.cross(r) + cosf(theta) * (r - r.dot(z) * z);
		}
		vert += linear_vel * dt;
	}

	void SphereRender::updateVertices()
	{
		// Map data
		Vec3f* verts;
		size_t num_bytes;
		cudaGraphicsMapResources(1, &m_cudaResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&verts, &num_bytes, m_cudaResource);

		updateSphereVerticesKernel <<< get_block_num(m_num_node), g_block_dim >>>
			(m_num_node, m_dt, m_origin, cast<float>(m_sphere->m_velocity), cast<float>(m_sphere->m_angular_velocity), verts);
		m_origin += m_dt * cast<float>(m_sphere->m_velocity);

		// Upmap data
		cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
	}

	void SphereRender::draw(const Shader* shader)
	{
		if (m_sphere->m_activate)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

			glPolygonMode(GL_FRONT, GL_FILL);
			shader->setVec4f("vertex_color", m_face_color, 0.5f);
			glDrawElements(GL_QUADS, 4 * m_num_quad, GL_UNSIGNED_INT, 0); // quad
			glDrawElements(GL_TRIANGLES, 3 * m_num_tri, GL_UNSIGNED_INT, (void*)(4 * m_num_quad * sizeof(unsigned)));  // tri

			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			shader->setVec4f("vertex_color", m_frame_color, 1.0f);
			glDrawElements(GL_QUADS, 4 * m_num_quad, GL_UNSIGNED_INT, 0); // quad
			glDrawElements(GL_TRIANGLES, 3 * m_num_tri, GL_UNSIGNED_INT, (void*)(4 * m_num_quad * sizeof(unsigned)));  // tri
		}
	}

	PlaneRender::PlaneRender(const ClothSim* cloth_sim, const Plane* plane) :
		ExternalObjectRender(cloth_sim), m_plane(plane)
	{
		// Init buffers
		glGenBuffers(1, &m_verticesBuffer);
		glGenBuffers(1, &m_indicesBuffer);

		// Buffer vertices data
		m_vertices.emplace_back(1.f, 0.f, 1.f);
		m_vertices.emplace_back(1.f, 0.f, -1.f);
		m_vertices.emplace_back(-1.f, 0.f, -1.f);
		m_vertices.emplace_back(-1.f, 0.f, 1.f);

		Eigen::Vec3f dir = cast_2_Eigen<float>(plane->m_direction);
		Eigen::Vec3f unit_y = Eigen::Vec3f::UnitY();
		Eigen::AngleAxisf rot(acos(unit_y.dot(dir)), unit_y.cross(dir));
		for (auto& v : m_vertices) v = rot * v * m_size + cast_2_Eigen<float>(plane->m_origin);

		glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(Eigen::Vec3f), m_vertices.data(), GL_DYNAMIC_DRAW); // 4 nodes

		// Buffer indices data
		unsigned h_idx[] = { 0, 1, 2, 3 };
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(h_idx), h_idx, GL_STATIC_DRAW); // 1 quad
	}

	PlaneRender::~PlaneRender()
	{
		glDeleteBuffers(1, &m_verticesBuffer);
		glDeleteBuffers(1, &m_indicesBuffer);
	}

	void PlaneRender::updateVertices()
	{
		for (Eigen::Vec3f& v : m_vertices)
			v += m_dt * cast_2_Eigen<float>(m_plane->m_velocity);

		glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(Vec3f), (void*)(m_vertices.data()), GL_DYNAMIC_DRAW);
	}

	void PlaneRender::draw(const Shader* shader)
	{
		if (m_plane->m_activate)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			shader->setVec4f("vertex_color", m_face_color, 0.5f);
			glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);
		}
	}
}