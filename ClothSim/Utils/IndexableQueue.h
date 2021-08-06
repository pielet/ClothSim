#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace cloth
{
	template <typename ScalarT>
	class IndexableQueue
	{
	public:
		IndexableQueue() : m_capacity(0), m_start(0), m_size(0), m_dim(0) {}
		~IndexableQueue()
		{
			for (auto& dp : m_queue)
			{
				if (dp) cudaFree(dp);
			}
		}

		void setCapacity(int n, int dim)
		{
			m_capacity = n;
			m_dim = dim;
			m_queue.resize(n, NULL);

			for (auto& dp : m_queue)
			{
				cudaMalloc((void**)&dp, dim * sizeof(ScalarT));
			}
		}

		void enqueue(const ScalarT* vec)
		{
			cudaMemcpy(m_queue[(m_start + m_size) % m_capacity], vec, m_dim * sizeof(ScalarT), cudaMemcpyDeviceToDevice);

			if (m_size < m_capacity - 1) ++m_size;
			else m_start = (m_start + 1) % m_capacity;
		}

		void empty()
		{
			m_start = m_size = 0;
		}

		const ScalarT* operator[](int i) const
		{
			assert(i >= 0 && i < m_size);

			return m_queue[(m_start + i) % m_capacity];
		}

		int size() const { return m_size; }

	protected:
		std::vector<ScalarT*> m_queue;

		int m_capacity;
		int m_dim; //< dimension of content vector

		int m_start;
		int m_size; // in [0, m_capacity - 1]
	};
}