#ifndef UTILITY_H
#define UTILITY_H

#include <Eigen/Core>
#include <chrono>
#include <ctime>
#include <cmath>

const float M_PI = 3.141592653589793238462643383279f;

typedef Eigen::Matrix<float, 3, 3> Mat3f;
typedef Eigen::Matrix<float, 4, 4> Mat4f;
typedef Eigen::Matrix<float, 3, 1> Vec3f;
typedef Eigen::Matrix<float, 4, 1> Vec4f;

class Timer
{
public:
	void start()
	{
		m_StartTime = std::chrono::system_clock::now();
		m_bRunning = true;
	}

	void stop()
	{
		m_EndTime = std::chrono::system_clock::now();
		m_bRunning = false;
	}

	double elapsedMilliseconds()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime;

		if (m_bRunning)
		{
			endTime = std::chrono::system_clock::now();
		}
		else
		{
			endTime = m_EndTime;
		}

		return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
	}

	double elapsedSeconds()
	{
		return elapsedMilliseconds() / 1000.0;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;
	std::chrono::time_point<std::chrono::system_clock> m_EndTime;
	bool m_bRunning = false;
};

#endif // !UTILITY_H
