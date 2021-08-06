#ifndef EXTERNAL_OBJECT_RENDER_H
#define EXTERNAL_OBJECT_RENDER_H

#include <vector>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include "../Utility.h"
#include "../../ClothSim/Utils/MathDef.h"
#include "../../ClothSim/ClothSim.h"
#include "../../ClothSim/ExternalCollision.h"

class Shader;

namespace cloth
{
	class ExternalObjectRender
	{
	public:
		ExternalObjectRender(const ClothSim* cloth_sim);
		virtual ~ExternalObjectRender() {}

		virtual void updateVertices() = 0;
		virtual void draw(const Shader* shader) = 0;

	protected:
		float m_dt;
		Eigen::Vec3f m_face_color;
		Eigen::Vec3f m_frame_color;
	};

	class SphereRender : public ExternalObjectRender
	{
	public:
		SphereRender(const ClothSim* cloth_sim, const Sphere* sphere);
		virtual ~SphereRender();

		virtual void updateVertices();
		virtual void draw(const Shader* shader);

	protected:
		Vec3f m_origin;
		const Sphere* m_sphere;

		//! Discretize
		int m_num_node;
		int m_num_tri;
		int m_num_quad;

		//! GL resource
		GLuint m_verticesBuffer;
		GLuint m_indicesBuffer;
		struct cudaGraphicsResource* m_cudaResource;
	};

	class PlaneRender : public ExternalObjectRender
	{
	public:
		PlaneRender(const ClothSim* cloth_sim, const Plane* plane);
		virtual ~PlaneRender();

		virtual void updateVertices();
		virtual void draw(const Shader* shader);

	protected:
		float m_size = 10.f;
		std::vector<Eigen::Vec3f> m_vertices;
		const Plane* m_plane;

		//! GL resource
		GLuint m_verticesBuffer;
		GLuint m_indicesBuffer;
	};
}

#endif