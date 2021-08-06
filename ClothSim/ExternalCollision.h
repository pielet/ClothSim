#pragma once

#include "Utils/MathDef.h"

namespace cloth
{
	struct ExternalCollisionInfo
	{
		int idx;
		Scalar hit_time;
		Vec3x normal;
		Vec3x hit_pos;
		Vec3x hit_vel;
	};

	class ExternalObject
	{
	public:
		ExternalObject(const Vec3x& origin);
		virtual ~ExternalObject() = default;

		Vec3x m_origin; //< use this to do CCD
		Vec3x m_velocity;
		Vec3x m_angular_velocity;

		bool m_activate;

		int m_group_idx;
	};

	
	class Sphere : public ExternalObject
	{
	public:
		Sphere(const Vec3x& origin, Scalar radius);
		virtual ~Sphere() = default;

		//! Perform CCD of point and sphere
		//! Assume *this will be passed by value to kernel functions
		CUDA_CALLABLE_MEMBER bool collisionDetection(Scalar h, const Vec3x& x, const Vec3x& v, ExternalCollisionInfo& info);

		Scalar m_radius;
	};

	
	class Plane : public ExternalObject
	{
	public:
		Plane(const Vec3x& origin, const Vec3x& dir);
		virtual ~Plane() = default;

		//! Perform CCD of point and plane
		//! Assume *this will be passed by value to kernel functions
		CUDA_CALLABLE_MEMBER bool collisionDetection(Scalar h, const Vec3x& x, const Vec3x& v, ExternalCollisionInfo& info);

		Vec3x m_direction;
	};
}