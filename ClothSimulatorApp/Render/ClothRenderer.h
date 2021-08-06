#ifndef CLOTH_RENDERER_H
#define CLOTH_RENDERER_H

#include <vector>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include "../../ClothSim/Utils/MathDef.h"
#include "../../ClothSim/ClothSim.h"
#include "ExternalObjectRender.h"

class Shader;

namespace cloth
{
	class ClothRenderer
	{
	public:
		ClothRenderer(const ClothSim* cloth_sim);
		~ClothRenderer();

		void draw(const Shader* shader);
		void ackGeometryChange() { m_geometryChanged = true; }

	protected:
		void updateVertices();

		int m_num_nodes;
		int m_num_faces;

		Eigen::Vec3f m_face_color;
		Eigen::Vec3f m_frame_color;

		const ClothSim* m_sim;

		std::vector<ExternalObjectRender*> m_external_renders;

		bool m_geometryChanged;

		GLuint m_verticesBuffer;
		GLuint m_indicesBuffer;

		struct cudaGraphicsResource* m_cudaResource;
	};
}

#endif // !CLOTH_RENDERER_H
