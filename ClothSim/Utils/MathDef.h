// Author: Shiyang Jia (jsy0325@foxmail.com)
// Data: 12/23/2020

#ifndef DEFINITION_H
#define DEFINITION_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define INLINE __forceinline__
#else
#define CUDA_CALLABLE_MEMBER
#define INLINE inline
#endif

typedef float Scalar;	// global scope

const int g_block_dim = 128;
#define get_block_num(n) (n + g_block_dim - 1) / g_block_dim

/*************************** Eigen definitions *********************************/
namespace Eigen
{
	typedef Matrix<float, 3, 1> Vec3f;
	typedef Matrix<float, 4, 1> Vec4f;
	typedef Matrix<float, Eigen::Dynamic, 1> VecXf;
	typedef Matrix<float, 4, 4> Mat4f;

	typedef Matrix<Scalar, 2, 1> Vec2x; ///< 2d scalar vector
	typedef Matrix<Scalar, 3, 1> Vec3x; ///< 3d scalar vector
	typedef Matrix<Scalar, 4, 1> Vec4x; ///< 4d scalar vector
	typedef Matrix<Scalar, Eigen::Dynamic, 1> VecXx; ///< arbitrary dimension scalar vector

	typedef Matrix<Scalar, 2, 2> Mat2x; ///< 2x2 scalar matrix
	typedef Matrix<Scalar, 3, 3> Mat3x; ///< 3x3 scalar matrix
	typedef Matrix<Scalar, 4, 4> Mat4x; ///< 4x4 scalar matrix
	typedef Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatXx; ///< arbitrary dimension scalar matrix

	typedef AngleAxis<Scalar> AngleAxisx;
	typedef Quaternion<Scalar> Quaternionx;
	typedef Translation<Scalar, 3> Translationx;
	typedef Transform<Scalar, 3, Affine> Transformx;
}

namespace cloth
{
	/************************ cuda callable vector and matrix definitions *************************/
	template <typename T, int n>
	class Vec
	{
	public:
		CUDA_CALLABLE_MEMBER Vec() {}
		CUDA_CALLABLE_MEMBER ~Vec() {}
		CUDA_CALLABLE_MEMBER Vec(int x, int y, int z);	// assert n == 3
		CUDA_CALLABLE_MEMBER Vec(int x, int y, int z, int w);	// assert n == 4

		CUDA_CALLABLE_MEMBER T& operator()(int i) { return value[i]; }
		CUDA_CALLABLE_MEMBER const T& operator()(int i) const { return value[i]; }

		CUDA_CALLABLE_MEMBER Vec<T, n> operator-(const Vec<T, n>& other) const;
		CUDA_CALLABLE_MEMBER Vec<T, n> operator+(const Vec<T, n>& other) const;
		CUDA_CALLABLE_MEMBER Vec<T, n> operator+(T a) const;
		CUDA_CALLABLE_MEMBER Vec<T, n>& operator+=(T a);
		CUDA_CALLABLE_MEMBER Vec<T, n>& operator+=(const Vec<T, n>& other);

		CUDA_CALLABLE_MEMBER void setZero();
		CUDA_CALLABLE_MEMBER T squareNorm() const;
		CUDA_CALLABLE_MEMBER T norm() const;
		CUDA_CALLABLE_MEMBER Vec<T, 3> cross(const Vec<T, 3>& other) const;
		CUDA_CALLABLE_MEMBER T dot(const Vec<T, n>& other) const;

		T value[n];
	};

	template <typename T, int m, int n>
	class Mat
	{
	public:
		CUDA_CALLABLE_MEMBER Mat() {}
		CUDA_CALLABLE_MEMBER ~Mat() {}

		CUDA_CALLABLE_MEMBER T& operator()(int i, int j) { return value[i][j]; }
		CUDA_CALLABLE_MEMBER const T& operator()(int i, int j) const { return value[i][j]; }

		CUDA_CALLABLE_MEMBER Vec<T, m> col(int i) const;
		CUDA_CALLABLE_MEMBER void setCol(int i, const Vec<T, m>& vec);
		CUDA_CALLABLE_MEMBER void setZero();

		CUDA_CALLABLE_MEMBER Mat<T, m, n> inverse() const;	// only used in Mat<Scalar, 2, 2>
		CUDA_CALLABLE_MEMBER Mat<T, n, m> transpose() const;
		CUDA_CALLABLE_MEMBER T trace() const;	// assert m == n

		CUDA_CALLABLE_MEMBER static Mat<T, m, n> Identity();	// assert m == n
		CUDA_CALLABLE_MEMBER static Mat<T, m, n> Zero();

		CUDA_CALLABLE_MEMBER Mat<T, m, n> operator+(const Mat<T, m, n>& other) const;
		CUDA_CALLABLE_MEMBER Mat<T, m, n>& operator+=(const Mat<T, m, n>& other);

		CUDA_CALLABLE_MEMBER Mat<T, m, n> operator*(T a);
		CUDA_CALLABLE_MEMBER Mat<T, m, n> operator/(T a);
		CUDA_CALLABLE_MEMBER Vec<T, m> operator*(const Vec<T, n>& vec);
		template <int p>
		CUDA_CALLABLE_MEMBER Mat<T, m, p> operator*(const Mat<T, n, p>& other);

		T value[m][n];
	};

	//template <typename T>
	//class Mat<T, 3, 3>
	//{
	//public:
	//	CUDA_CALLABLE_MEMBER Mat() {}
	//	CUDA_CALLABLE_MEMBER ~Mat() {}
	//	CUDA_CALLABLE_MEMBER T& operator()(int i, int j) { return value[i][j]; }
	//	CUDA_CALLABLE_MEMBER const T& operator()(int i, int j) const { return value[i][j]; }
	//private:
	//	T value[3][3];
	//};

	/***************************** Implementation *****************************/
	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, n>::Vec(int x, int y, int z)
	{
		assert(n == 3);
		value[0] = x; value[1] = y; value[2] = z;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, n>::Vec(int x, int y, int z, int w)
	{
		assert(n == 4);
		value[0] = x; value[1] = y; value[2] = z; value[3] = w;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, n> Vec<T, n>::operator+(const Vec<T, n>& other) const
	{
		Vec<T, n> vec;
#pragma unroll
		for (int i = 0; i < n; ++i) vec.value[i] = value[i] + other.value[i];
		return vec;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, n> Vec<T, n>::operator+(T a) const
	{
		Vec<T, n> vec;
#pragma unroll
		for (int i = 0; i < n; ++i) vec.value[i] = value[i] + a;
		return vec;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, n> Vec<T, n>::operator-(const Vec<T, n>& other) const
	{
		Vec<T, n> vec;
#pragma unroll
		for (int i = 0; i < n; ++i) vec.value[i] = value[i] - other.value[i];
		return vec;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, n>& Vec<T, n>::operator+=(T a)
	{
#pragma unroll
		for (int i = 0; i < n; ++i) value[i] += a;
		return *this;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, n>& Vec<T, n>::operator+=(const Vec<T, n>& other)
	{
#pragma unroll
		for (int i = 0; i < n; ++i) value[i] += other.value[i];
		return *this;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER void Vec<T, n>::setZero()
	{
#pragma unroll
		for (int i = 0; i < n; ++i) value[i] = 0;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER T Vec<T, n>::squareNorm() const
	{
		T norm2 = 0;
#pragma unroll
		for (int i = 0; i < n; ++i) norm2 += value[i] * value[i];
		return norm2;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER T Vec<T, n>::norm() const
	{
		return sqrt(squareNorm());
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER Vec<T, 3> Vec<T, n>::cross(const Vec<T, 3>& other) const
	{
		assert(n == 3);

		Vec<T, 3> vec;
		vec.value[0] = value[1] * other.value[2] - value[2] * other.value[1];
		vec.value[1] = value[2] * other.value[0] - value[0] * other.value[2];
		vec.value[2] = value[0] * other.value[1] - value[1] * other.value[0];
		return vec;
	}

	template <typename T, int n>
	CUDA_CALLABLE_MEMBER T Vec<T, n>::dot(const Vec<T, n>& other) const
	{
		T res = 0;
#pragma unroll
		for (int i = 0; i < n; ++i) res += other.value[i] * value[i];
		return res;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Vec<T, m> Mat<T, m, n>::col(int ci) const
	{
		Vec<T, m> vec;
#pragma unroll
		for (int i = 0; i < m; ++i) vec.value[i] = value[i][ci];
		return vec;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER void Mat<T, m, n>::setCol(int ci, const Vec<T, m>& vec)
	{
#pragma unroll
		for (int i = 0; i < m; ++i) value[i][ci] = vec.value[i];
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER void Mat<T, m, n>::setZero()
	{
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < n; ++j)
				value[i][j] = 0;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, n, m> Mat<T, m, n>::transpose() const
	{
		assert(m == n);
		Mat<T, n, m> mat;
#pragma unroll
		for (int i = 0; i < n; ++i)
#pragma unroll 
			for (int j = 0; j < m; ++j)
				mat.value[i][j] = value[j][i];
		return mat;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER T Mat<T, m, n>::trace() const
	{
		assert(m == n);
		T res = 0;
#pragma unroll
		for (int i = 0; i < m; ++i) res += value[i][i];
		return res;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, m, n> Mat<T, m, n>::Identity()
	{
		assert(m == n);
		Mat<T, m, n> mat; mat.setZero();
#pragma unroll
		for (int i = 0; i < m; ++i) mat.value[i][i] = 1;
		return mat;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, m, n> Mat<T, m, n>::Zero()
	{
		Mat<T, m, n> mat;
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < n; ++j)
				mat.value[i][j] = 0;
		return mat;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, m, n> Mat<T, m, n>::operator+(const Mat<T, m, n>& other) const
	{
		Mat<T, m, n> mat;
#pragma unroll
		for (int i = 0; i < m; ++i) 
#pragma unroll
			for (int j = 0; j < n; ++j)
				mat.value[i][j] = value[i][j] + other.value[i][j];
		return mat;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, m, n>& Mat<T, m, n>::operator+=(const Mat<T, m, n>& other)
	{
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < n; ++j)
				value[i][j] += other.value[i][j];
		return *this;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, m, n> Mat<T, m, n>::operator*(T a)
	{
		Mat<T, m, n> mat;
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < n; ++j)
				mat.value[i][j] = a * value[i][j];
		return mat;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, m, n> Mat<T, m, n>::operator/(T a)
	{
		Mat<T, m, n> mat;
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < n; ++j)
				mat.value[i][j] = value[i][j] / a;
		return mat;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Mat<T, m, n> operator*(T a, const Mat<T, m, n>& in)
	{
		Mat<T, m, n> out;
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < n; ++j)
				out.value[i][j] = a * in.value[i][j];
		return out;
	}

	template <typename T, int m, int n>
	CUDA_CALLABLE_MEMBER Vec<T, m> Mat<T, m, n>::operator*(const Vec<T, n>& in)
	{
		Vec<T, m> out; out.setZero();
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < n; ++j)
				out.value[i] += value[i][j] * in.value[j];
		return out;
	}

	template <typename T, int m, int n> template <int p>
	CUDA_CALLABLE_MEMBER Mat<T, m, p> Mat<T, m, n>::operator*(const Mat<T, n, p>& in)
	{
		Mat<T, m, p> out; out.setZero();
#pragma unroll
		for (int i = 0; i < m; ++i)
#pragma unroll
			for (int j = 0; j < p; ++j)
#pragma unroll
				for (int k = 0; k < n; ++k)
					out.value[i][j] += value[i][k] & in.value[k][j];
		return out;
	}

	template <>
	CUDA_CALLABLE_MEMBER INLINE Mat<Scalar, 2, 2> Mat<Scalar, 2, 2>::inverse() const
	{
		Mat<Scalar, 2, 2> out;
		Scalar a = value[0][0], b = value[0][1], c = value[1][0], d = value[1][1];
		out.value[0][0] = d;
		out.value[0][1] = -b;
		out.value[1][0] = -c;
		out.value[1][1] = a;
		out = out / (a * d - b * c);
		return out;
	}

	typedef Vec<int, 3> FaceIdx;
	typedef Vec<int, 4> EdgeIdx;

	typedef Vec<Scalar, 2> Vec2x;
	typedef Vec<Scalar, 3> Vec3x;
	typedef Vec<Scalar, 4> Vec4x;
	typedef Mat<Scalar, 2, 2> Mat2x;
	typedef Mat<Scalar, 3, 3> Mat3x;
}

#endif // !DEFINITION_H