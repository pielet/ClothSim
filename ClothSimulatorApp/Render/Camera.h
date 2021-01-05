#ifndef CAMERA_H
#define CAMERA_H

#include "../Utility.h"

class Camera
{
public:
	enum Motion {SCALE, ROTATE, TRANSLATE};

	Camera(int w, int h);
	~Camera() {};

	void setPose(const Vec3f& eye, const Vec3f& lookAt);
	void setViewPort(int w, int h);
	void setZClipping(float near, float far);

	void beginMotion(Motion motion, int x, int y);
	void move(int x, int y);
	void scroll(int direction);

	Mat4f getViewMatrix() const;
	Mat4f getPerspectiveMatrix() const;

protected:
	Vec3f m_eye;
	Vec3f m_lookAt;
	Vec3f m_right;
	Vec3f m_up;

	Motion m_motion;
	int m_x;
	int m_y;

	int m_width, m_height;
	float m_near, m_far;
};

#endif // !CAMERA_H
