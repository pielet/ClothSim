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

		//! Update velocities according to selected motion script
		virtual void setVelocity(const Vec3x& vel);
		//! Update origin
		virtual inline void update(Scalar h);

	protected:
		Vec3x m_origin;
		Vec3x m_velocity;

		int m_group_idx;
	};

	
	class Sphere : public ExternalObject
	{
	public:
		Sphere(const Vec3x& origin, Scalar radius);
		virtual ~Sphere() = default;

		//virtual void setVelocity(const Vec3x& vel);

		//! Perform CCD of point and sphere
		//! Assume *this will be passed by value to kernel functions
		CUDA_CALLABLE_MEMBER bool collisionDetection(Scalar h, const Vec3x& x, const Vec3x& v, ExternalCollisionInfo& info);

	protected:
		Scalar m_radius;
		Vec3x m_angular_velocity;
	};

	
	class Plane : public ExternalObject
	{
	public:
		Plane(const Vec3x& origin, const Vec3x& dir);
		virtual ~Plane() = default;

		//virtual void setVelocity(const Vec3x& vel);

		//! Perform CCD of point and plane
		//! Assume *this will be passed by value to kernel functions
		CUDA_CALLABLE_MEMBER bool collisionDetection(Scalar h, const Vec3x& x, const Vec3x& v, ExternalCollisionInfo& info);

	protected:
		Vec3x m_direction;
	};
}