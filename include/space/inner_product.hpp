#pragma once

#include <immintrin.h>

#include <cstddef>

namespace symqg::space {

    inline float ip_sim(
            const float* __restrict__ vec0, const float* __restrict__ vec1, size_t dim
    ) {
        float result = 0;
#if defined(__AVX512F__)
        size_t mul16 = dim - (dim & 0b1111);
        auto sum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i < mul16; i += 16) {
            auto xxx = _mm512_loadu_ps(&vec0[i]);
            auto yyy = _mm512_loadu_ps(&vec1[i]);
            sum = _mm512_fmadd_ps(xxx, yyy, sum);
        }
        result = _mm512_reduce_add_ps(sum);
        for (; i < dim; ++i) {
            result += vec0[i]  * vec1[i];
        }

#elif defined(__AVX2__)
        size_t mul8 = dim - (dim & 0b111);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i < mul8; i += 8) {
        __m256 xx = _mm256_loadu_ps(&vec0[i]);
        __m256 yy = _mm256_loadu_ps(&vec1[i]);
        sum = _mm256_fmadd_ps(t, t, sum);
    }
    result = reduce_add_m256(sum);
    for (; i < dim; ++i) {
        result += vec0[i]  * vec1[i];
    }

#else
    for (size_t i = 0; i < dim; ++i) {
        result +=  vec0[i] * vec1[i]
    }
#endif
        return result;
    }


}  // namespace symqg::space