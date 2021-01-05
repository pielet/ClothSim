#ifndef CLOTH_RENDERER_H
#define CLOTH_RENDERER_H

#include <vector>
#include <GL/glew.h>
#include <cuda_runtime.h>

class Shader;
namespace cloth { class ClothSim; }

class ClothRenderer
{
public:
	ClothRenderer(int num_nodes, int num_faces, const cloth::ClothSim* cloth_sim);
	~ClothRenderer();

	void draw(const Shader* shader);
	void ackGeometryChange() { m_geometryChanged = true; }

protected:
	void updateVertices();

	int m_num_nodes;
	int m_num_faces;

	const cloth::ClothSim* m_sim;

	bool m_geometryChanged;

	GLuint m_verticesBuffer;
	GLuint m_indicesBuffer;

	struct cudaGraphicsResource* m_cudaResource;
};

#endif // !CLOTH_RENDERER_H
