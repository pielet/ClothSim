#include "ClothRenderer.h"
#include <cuda_gl_interop.h>
#include "Shader.h"
#include "../../ClothSim/ExternalCollision.h"

namespace cloth
{
	__global__ void copyIndices(int n_faces, int offset, unsigned* dst, const FaceIdx* src)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n_faces) return;

		FaceIdx idx = src[i] + offset;
		dst[3 * i] = idx(0);
		dst[3 * i + 1] = idx(1);
		dst[3 * i + 2] = idx(2);
	}

	__global__ void copyPositions(int n_nodes, float* dst, const Vec3x* src)
	{
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= n_nodes) return;

		dst[3 * idx] = src[idx](0);
		dst[3 * idx + 1] = src[idx](1);
		dst[3 * idx + 2] = src[idx](2);
	}

	ClothRenderer::ClothRenderer(const ClothSim* cloth_sim) :
		m_sim(cloth_sim),
		m_geometryChanged(false)
	{
		m_num_nodes = cloth_sim->getNumTotalNodes();
		m_num_faces = cloth_sim->getNumTotalFaces();

		m_face_color = Eigen::Vec3f(255.f, 153.f, 102.f) / 255.f;
		m_frame_color = Eigen::Vec3f(77.f, 26.f, 0.f) / 255.f;

		// Init buffers
		glGenBuffers(1, &m_verticesBuffer);
		glGenBuffers(1, &m_indicesBuffer);

		// Generate buffer
		glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
		glBufferData(GL_ARRAY_BUFFER, 3 * m_num_nodes * sizeof(float), NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * m_num_faces * sizeof(unsigned), NULL, GL_STATIC_DRAW);

		// Buffer indices data
		cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_indicesBuffer, cudaGraphicsMapFlagsWriteDiscard);
		unsigned* idx;
		size_t num_bytes;
		cudaGraphicsMapResources(1, &m_cudaResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&idx, &num_bytes, m_cudaResource);
		int face_count = 0;
		for (int i = 0; i < cloth_sim->getNumCloths(); ++i)
		{
			copyIndices <<< get_block_num(m_num_faces), g_block_dim >>> (
				cloth_sim->getNumFaces(i), cloth_sim->getOffset(i), &idx[3 * face_count], cloth_sim->getFaceIndices(i));
			face_count += cloth_sim->getNumFaces(i);
		}
		cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
		cudaGraphicsUnregisterResource(m_cudaResource);

		// Register CUDA resources
		cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_verticesBuffer, cudaGraphicsMapFlagsWriteDiscard);
		updateVertices();

		// Construct external renders
		for (const ExternalObject* op : cloth_sim->getExternalObjects())
		{
			if (const Sphere* sp = dynamic_cast<const Sphere*>(op))
				m_external_renders.push_back(new SphereRender(cloth_sim, sp));
			else if (const Plane* pp = dynamic_cast<const Plane*>(op))
				m_external_renders.push_back(new PlaneRender(cloth_sim, pp));
		}
	}

	ClothRenderer::~ClothRenderer()
	{
		cudaGraphicsUnregisterResource(m_cudaResource);

		glDeleteBuffers(1, &m_verticesBuffer);
		glDeleteBuffers(1, &m_indicesBuffer);

		for (ExternalObjectRender* rp : m_external_renders)
			delete rp;
	}

	void ClothRenderer::draw(const Shader* shader)
	{
		if (m_geometryChanged)
		{
			updateVertices();

			for (ExternalObjectRender* rp : m_external_renders)
				rp->updateVertices();

			m_geometryChanged = false;
		}

		glEnableVertexAttribArray(0);

		// Draw cloth meshs
		glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

		// red triangles
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		shader->setVec4f("vertex_color", m_face_color, 1.f);
		glDrawElements(GL_TRIANGLES, 3 * m_num_faces, GL_UNSIGNED_INT, 0);

		// black wireframe
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		shader->setVec4f("vertex_color", m_frame_color, 1.f);
		glDrawElements(GL_TRIANGLES, 3 * m_num_faces, GL_UNSIGNED_INT, 0);

		for (ExternalObjectRender* rp : m_external_renders)
			rp->draw(shader);
	}

	void ClothRenderer::updateVertices()
	{
		// Map data
		float* ptr;
		size_t num_bytes;
		cudaGraphicsMapResources(1, &m_cudaResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, m_cudaResource);

		copyPositions <<< get_block_num(m_num_nodes), g_block_dim >>> (m_num_nodes, ptr, m_sim->getPositions());

		// Upmap data
		cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
	}
}