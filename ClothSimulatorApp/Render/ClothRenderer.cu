#include "ClothRenderer.h"
#include <vector>
#include <iostream>
#include <cuda_gl_interop.h>
#include "Shader.h"
#include "../../ClothSim/Utils/MathDef.h"
#include "../../ClothSim/ClothSim.h"

__global__ void copyIndices(int n_faces, int offset, unsigned* dst, const cloth::FaceIdx* src)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n_faces) return;

	cloth::FaceIdx idx = src[i] + offset;
	dst[3 * i] = idx(0);
	dst[3 * i + 1] = idx(1);
	dst[3 * i + 2] = idx(2);
}

__global__ void copyPositions(int n_nodes, float* dst, const cloth::Vec3x* src)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= n_nodes) return;

	dst[3 * idx] = src[idx](0);
	dst[3 * idx + 1] = src[idx](1);
	dst[3 * idx + 2] = src[idx](2);
}

ClothRenderer::ClothRenderer(int num_nodes, int num_faces, const cloth::ClothSim* cloth_sim) :
	m_num_nodes(num_nodes),
	m_num_faces(num_faces),
	m_sim(cloth_sim),
	m_geometryChanged(true)
{
	// Init buffers
	glGenBuffers(1, &m_verticesBuffer);
	glGenBuffers(1, &m_indicesBuffer);

	// Generate buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * num_nodes * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * num_faces * sizeof(unsigned), NULL, GL_STATIC_DRAW);

	// Buffer indices data
	cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_indicesBuffer, cudaGraphicsMapFlagsWriteDiscard);
	unsigned* idx;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &m_cudaResource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&idx, &num_bytes, m_cudaResource);
	int face_count = 0;
	for (int i = 0; i < cloth_sim->getNumCloths(); ++i)
	{
		copyIndices <<< cloth::get_block_num(num_faces), cloth::g_block_dim >>> (
			cloth_sim->getNumFaces(i), cloth_sim->getOffset(i), &idx[3 * face_count], cloth_sim->getFaceIndices(i));
		face_count += cloth_sim->getNumFaces(i);
	}
	cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
	cudaGraphicsUnregisterResource(m_cudaResource);

	// Register CUDA resources
	cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_verticesBuffer, cudaGraphicsMapFlagsWriteDiscard);
}

ClothRenderer::~ClothRenderer()
{
	cudaGraphicsUnregisterResource(m_cudaResource);

	glDeleteBuffers(1, &m_verticesBuffer);
	glDeleteBuffers(1, &m_indicesBuffer);
}

void ClothRenderer::draw(const Shader* shader)
{
	if (m_geometryChanged)
	{
		updateVertices();
		m_geometryChanged = false;
	}

	glEnableVertexAttribArray(0);

	// Draw cloth meshs
	glBindBuffer(GL_ARRAY_BUFFER, m_verticesBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

	// red triangles
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	shader->setVec4f("vertex_color", 255.f / 255.f, 153.f / 255.f, 102.f / 255.f, 1.f);
	glDrawElements(GL_TRIANGLES, 3 * m_num_faces, GL_UNSIGNED_INT, 0);

	// black wireframe
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	shader->setVec4f("vertex_color", 77.f / 255.f, 26.f / 255.f, 0.f / 255.f, 1.f);
	glDrawElements(GL_TRIANGLES, 3 * m_num_faces, GL_UNSIGNED_INT, 0);
}

void ClothRenderer::updateVertices()
{
	// Map data
	float* ptr;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &m_cudaResource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& ptr, &num_bytes, m_cudaResource);

	copyPositions <<< cloth::get_block_num(m_num_nodes), cloth::g_block_dim >>> (m_num_nodes, ptr, m_sim->getPositions());

	// Upmap data
	cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
}
