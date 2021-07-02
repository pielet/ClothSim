#include "ExternalCollision.h"
#include "Utils/MathUtility.h"

namespace cloth
{
	ExternalObject::ExternalObject(const Vec3x& origin) 
		: m_origin(origin), m_velocity(0.f, 0.f, 0.f)
	{ }

	void ExternalObject::setVelocity(const Vec3x& vel)
	{
		m_velocity = vel;
	}

	void ExternalObject::update(Scalar h)
	{
		m_origin += m_velocity * h;
	}

	Sphere::Sphere(const Vec3x& origin, Scalar radius)
		: ExternalObject(origin), m_radius(radius), m_angular_velocity(0.f, 0.f, 0.f)
	{ }

	CUDA_CALLABLE_MEMBER bool Sphere::collisionDetection(Scalar h, const Vec3x& x, const Vec3x& v, ExternalCollisionInfo& info)
	{
		Scalar a = (v - m_velocity).squareNorm();
		Scalar b = 2 * (x - m_origin).dot(v - m_velocity);
		Scalar c = (x - m_origin).squareNorm() - m_radius * m_radius;

		Scalar t0 = -1.f, t1 = -1.f;
		int n_root = solveQuadratic(a, b, c, t0, t1);

		if (t0 < 0 && t1 < 0) return false;

		Scalar t = fmin(t0, t1);
		if (t < 0) t = fmax(t0, t1);
		if (t > h) return false;

		info.hit_time = t;
		info.hit_pos = x + t * v;
		info.normal = (info.hit_pos - m_origin).normalized();
		info.hit_vel = m_velocity + m_angular_velocity.cross(info.hit_pos - m_origin);

		return true;
	}

	Plane::Plane(const Vec3x& origin, const Vec3x& dir)
		: ExternalObject(origin)
	{
		m_direction = dir.normalized();
	}

	CUDA_CALLABLE_MEMBER bool Plane::collisionDetection(Scalar h, const Vec3x& x, const Vec3x& v, ExternalCollisionInfo& info)
	{
		Scalar t = -(x - m_origin).dot(m_direction) / (v - m_velocity).dot(m_direction);
		 
		if (t < 0 || t > h) return false;

		info.hit_time = t;
		info.hit_pos = x + t * v;
		info.normal = m_direction;
		info.hit_vel = m_velocity;

		return true;
	}
}