#pragma once

#include "MathDef.h"

namespace cloth
{
	INLINE CUDA_MEMBER Scalar square(Scalar x)
	{
		return x * x;
	}

	INLINE CUDA_CALLABLE_MEMBER int solveQuadratic(Scalar a, Scalar b, Scalar c, Scalar& x1, Scalar& x2)
	{
		Scalar delta = b * b - 4 * a * c;

		if (delta < 0) return 0;
		else if (delta < EPS)
		{
			x1 = -b / (2 * a);
			return 1;
		}
		else
		{
			Scalar sqrt_delta = sqrt(delta);
			x1 = (-b + sqrt_delta) / (2 * a);
			x2 = (-b - sqrt_delta) / (2 * a);
			return 2;
		}
	}

	// Compute (a^2 + b^2)^(1/2) without destructive underflow or overflow
	INLINE CUDA_MEMBER Scalar pythag(Scalar a, Scalar b)
	{
		Scalar absa = fabs(a), absb = fabs(b);
		return (absa > absb ? absa * sqrt(1.0f + square(absb / absa)) :
			(absb == 0.0f ? 0.0f : absb * sqrt(1.0f + square(absa / absb))));
	}

	// SVD decomposition for square matrix
	// From http://numerical.recipes/ 2.6
	template <int n>
	CUDA_MEMBER bool SVDdecomp(Mat<Scalar, n, n>& U, Mat<Scalar, n, n>& V, Vec<Scalar, n>& sigma)
	{
		bool flag;
		int i, its, j, jj, k, l, nm;
		Scalar anorm, c, f, g, h, s, scale, x, y, z;
		Vec<Scalar, n> rv1;

		// Householder reduction to bidiagonal form
		g = scale = anorm = 0.0f;
		for (i = 0; i < n; i++) {
			l = i + 2;
			rv1(i) = scale * g;
			g = s = scale = 0.0f;

			for (k = i; k < n; k++) scale += fabs(U(k, i));

			if (fabs(scale) > EPS) {
				for (k = i; k < n; k++) {
					U(k, i) /= scale;
					s += U(k, i) * U(k, i);
				}
				f = U(i, i);
				g = -copysignf(sqrt(s), f);
				h = f * g - s;
				U(i, i) = f - g;
				for (j = l - 1; j < n; j++) {
					for (s = 0.0f, k = i; k < n; k++) s += U(k, i) * U(k, j);
					f = s / h;
					for (k = i; k < n; k++) U(k, j) += f * U(k, i);
				}
				for (k = i; k < n; k++) U(k, i) *= scale;
			}

			sigma(i) = scale * g;

			g = s = scale = 0.0f;
			if (i + 1 != n) {
				for (k = l - 1; k < n; k++) scale += fabs(U(i, k));
				if (fabs(scale) > EPS) {
					for (k = l - 1; k < n; k++) {
						U(i, k) /= scale;
						s += U(i, k) * U(i, k);
					}
					f = U(i, l - 1);
					g = -copysignf(sqrt(s), f);
					h = f * g - s;
					U(i, l - 1) = f - g;
					for (k = l - 1; k < n; k++) rv1(k) = U(i, k) / h;
					for (j = l - 1; j < n; j++) {
						for (s = 0.0f, k = l - 1; k < n; k++) s += U(j, k) * U(i, k);
						for (k = l - 1; k < n; k++) U(j, k) += s * rv1(k);
					}
					for (k = l - 1; k < n; k++) U(i, k) *= scale;
				}
			}
			anorm = fmax(anorm, (fabs(sigma(i)) + fabs(rv1(i))));
		}

		// Accumulation of right-hand transformations
		for (i = n - 1; i >= 0; i--) {
				if (i < n - 1) {
					if (fabs(g) > EPS) {
						for (j = l; j < n; j++) // Double division to avoid possible underflow
							V(j, i) = U(i, j) / U(i, l) / g;
						for (j = l; j < n; j++) {
							for (s = 0.0f, k = l; k < n; k++) s += U(i, k) * V(k, j);
							for (k = l; k < n; k++) V(k, j) += s * V(k, i);
						}
					}
					for (j = l; j < n; j++) V(i, j) = V(j, i) = 0.0f;
				}
			V(i, i) = 1.0;
			g = rv1(i);
			l = i;
		}

		// Accumulation of left-hand transformations
		for (i = n - 1; i >= 0; i--) {
			l = i + 1;
			g = sigma(i);
			for (j = l; j < n; j++) U(i, j) = 0.0f;
			if (fabs(g) > EPS) {
				g = 1.0f / g;
				for (j = l; j < n; j++) {
					for (s = 0.0f, k = l; k < n; k++) s += U(k, i) * U(k, j);
					f = (s / U(i, i)) * g;
					for (k = i; k < n; k++) U(k, j) += f * U(k, i);
				}
				for (j = i; j < n; j++) U(j, i) *= g;
			}
			else for (j = i; j < n; j++) U(j, i) = 0.0f;
			++U(i, i);
		}

		// Diagonalization of the bidiagonal form : Loop over
		for (k = n - 1; k >= 0; k--) {
			for (its = 0; its < 30; its++) { // singular values, and over allowed iterations.
				flag = true;
				for (l = k; l >= 0; l--) { // Test for splitting.
					nm = l - 1;
					if (l == 0 || fabs(rv1(l)) <= EPS * anorm) {
						flag = false;
						break;
					}
					if (fabs(sigma(nm)) <= EPS * anorm) break;
				}
				if (flag) {
					c = 0.0f; // Cancellation of rv1[l], if l > 0.
					s = 1.0f;
					for (i = l; i < k + 1; i++) {
						f = s * rv1(i);
						rv1(i) = c * rv1(i);
						if (fabs(f) <= EPS * anorm) break;
						g = sigma(i);
						h = pythag(f, g);
						sigma(i) = h;
						h = 1.0f / h;
						c = g * h;
						s = -f * h;
						for (j = 0; j < n; j++) {
							y = U(j, nm);
							z = U(j, i);
							U(j, nm) = y * c + z * s;
							U(j, i) = z * c - y * s;
						}
					}
				}

				z = sigma(k);
				if (l == k) { // Convergence.
					if (z < 0.0f) { // Singular value is made nonnegative.
						sigma(k) = -z;
						for (j = 0; j < n; j++) V(j, k) = -V(j, k);
					}
					break;
				}
				if (its == 29) return false;

				// Shift from bottom 2-by-2 minor
				x = sigma(l);
				nm = k - 1;
				y = sigma(nm);
				g = rv1(nm);
				h = rv1(k);
				f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0f * h * y);
				g = pythag(f, 1.0f);
				f = ((x - z) * (x + z) + h * ((y / (f + copysignf(g, f))) - h)) / x;
				c = s = 1.0f;

				// Next QR transformation
				for (j = l; j <= nm; j++) {
					i = j + 1;
					g = rv1(i);
					y = sigma(i);
					h = s * g;
					g = c * g;
					z = pythag(f, h);
					rv1(j) = z;
					c = f / z;
					s = h / z;
					f = x * c + g * s;
					g = g * c - x * s;
					h = y * s;
					y *= c;
					for (jj = 0; jj < n; jj++) {
						x = V(jj, j);
						z = V(jj, i);
						V(jj, j) = x * c + z * s;
						V(jj, i) = z * c - x * s;
					}
					z = pythag(f, h);
					sigma(j) = z; // Rotation can be arbitrary if z = 0.
					if (z) {
						z = 1.0f / z;
						c = f * z;
						s = h * z;
					}
					f = c * g + s * y;
					x = c * y - s * g;
					for (jj = 0; jj < n; jj++) {
						y = U(jj, j);
						z = U(jj, i);
						U(jj, j) = y * c + z * s;
						U(jj, i) = z * c - y * s;
					}
				}
				rv1(l) = 0.0f;
				rv1(k) = f;
				sigma(k) = x;
			}
		}

		return true;
	}

	// Given the output of decompose, this routine sorts the singular values, and corresponding columns
	// of U and V, by decreasing magnitude. Also, signs of corresponding columns are flipped so as to
	// maximize the number of positive elements.
	template <int n>
	CUDA_MEMBER void SVDreorder(Mat<Scalar, n, n>& U, Mat<Scalar, n, n>& V, Vec<Scalar, n>& sigma)
	{
		int i, j, k, s, inc = 1;
		Scalar sw;
		Vec<Scalar, n> su, sv;

		// Sort.The method is Shell¡¯s sort. 
		// (The work is negligible as compared to that already done in decompose.)
		do { inc *= 3; inc++; } while (inc <= n);
		do {
			inc /= 3;
			for (i = inc; i < n; i++) {
				sw = sigma(i);
				for (k = 0; k < n; k++) {
					su(k) = U(k, i);
					sv(k) = V(k, i);
				}
				j = i;
				while (sigma(j - inc) < sw) {
					sigma(j) = sigma(j - inc);
					for (k = 0; k < n; k++) {
						U(k, j) = U(k, j - inc);
						V(k, j) = V(k, j - inc);
					}
					j -= inc;
					if (j < inc) break;
				}
				sigma(j) = sw;
				for (k = 0; k < n; k++) {
					U(k, j) = su(k);
					V(k, j) = sv(k);
				}
			}
		} while (inc > 1);

		// Flip signs
		for (k = 0; k < n; k++) {
			s = 0;
			for (i = 0; i < n; i++) {
				if (U(i, k) < 0.) s++;
				if (V(i, k) < 0.) s++;
			}
			if (s > n) {
				for (i = 0; i < n; i++) {
					U(i, k) = -U(i, k);
					V(i, k) = -V(i, k);
				}
			}
		}
	}

}