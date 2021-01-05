#include "Camera.h"
#include <cmath>
#include <Eigen/Geometry>

const float mouse_speed = 1.f;
const float wheel_speed = 0.01f;

float degToRad(float deg)
{
	return deg / 180.f * M_PI;
}

Mat4f perspective(float fovy, float aspect, float zNear, float zFar)
{
	assert(aspect > 0);
	assert(zFar > zNear);

	float radf = degToRad(fovy);

	float tanHalfFovy = tan(radf / 2.0);
	Mat4f res = Mat4f::Zero();
	res(0, 0) = 1.0f / (aspect * tanHalfFovy);
	res(1, 1) = 1.0f / (tanHalfFovy);
	res(2, 2) = -(zFar + zNear) / (zFar - zNear);
	res(3, 2) = -1.0f;
	res(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);

	return res;
}

Mat4f lookAt(const Vec3f& eye, const Vec3f& center, const Vec3f& up)
{
	Vec3f f = (center - eye).normalized();
	Vec3f u = up.normalized();
	Vec3f s = f.cross(u).normalized();
	u = s.cross(f);

	Mat4f res;
	res << s.x(),  s.y(),  s.z(), -s.dot(eye),
		   u.x(),  u.y(),  u.z(), -u.dot(eye),
		  -f.x(), -f.y(), -f.z(),  f.dot(eye),
		  0, 0, 0, 1;

	return res;
}

Camera::Camera(int w, int h):
	m_width(w), m_height(h), m_near(0.01f), m_far(100.f)
{
	setPose(Vec3f(0.0f, 0.0f, 5.0f), Vec3f::Zero());
}

void Camera::setPose(const Vec3f& eye, const Vec3f& lookAt)
{
	m_eye = eye;
	m_lookAt = lookAt;
	Vec3f dir = (lookAt - eye).normalized();
	m_right = dir.cross(Vec3f(0, 1, 0)).normalized();
	m_up = m_right.cross(dir);
}

void Camera::setViewPort(int w, int h)
{
	m_width = w; m_height = h;
}

void Camera::setZClipping(float near, float far)
{
	m_near = near; m_far = far;
}

void Camera::beginMotion(Motion motion, int x, int y)
{
	m_motion = motion;
	m_x = x;
	m_y = y;
}

void Camera::move(int x, int y)
{
	float alpha = mouse_speed * ((m_lookAt - m_eye).norm() + 1.f);
	float dx = float(x - m_x) / m_width;
	float dy = float(y - m_y) / m_height;

	switch (m_motion)
	{
	case Camera::SCALE:
	{
		Vec3f dir = m_up.cross(m_right).normalized();
		m_eye -= alpha * dy * dir;
		break;
	}
	case Camera::ROTATE:
	{
		float x_angle = mouse_speed * dx;
		float y_angle = mouse_speed * dy;
		Vec3f new_dir = Eigen::AngleAxis<float>(-x_angle, m_up) * Eigen::AngleAxis<float>(-y_angle, m_right) * (m_eye - m_lookAt);
		m_eye = m_lookAt + new_dir;
		m_right = Eigen::AngleAxis<float>(-x_angle, m_up) * m_right;
		m_up = m_right.cross(-new_dir).normalized();
		break;
	}
	case Camera::TRANSLATE:
	{
		Vec3f delta_p = dx * m_right - dy * m_up;
		m_eye -= alpha * delta_p;
		m_lookAt -= alpha * delta_p;
		break;
	}
	}

	m_x = x;
	m_y = y;
}

void Camera::scroll(int direction)
{
	float alpha = wheel_speed * ((m_lookAt - m_eye).norm() + 1.f);
	m_eye += direction * alpha * m_up.cross(m_right).normalized();
}

Mat4f Camera::getViewMatrix() const
{
	return lookAt(m_eye, m_lookAt, m_up);
}

Mat4f Camera::getPerspectiveMatrix() const
{
	return perspective(60.f, float(m_width) / m_height, m_near, m_far);
}
