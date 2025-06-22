#pragma once

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <vector>

#include "../common.hpp"
#include "../utils/tools.hpp"

namespace symqg::space {

inline PID exact_nn(
    const float* data,
    const float* query,
    size_t num_points,
    size_t dim,
    size_t num_threads,
    const DistFunc<float>& dist_func_
) {
    std::vector<Candidate<float>> best_entries(num_threads, Candidate(0, FLT_MAX));

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        Candidate<float>& cur_entry = best_entries[tid];
        const float* cur_data = data + (dim * i);

        float distance = dist_func_(cur_data, query, dim);
        if (distance < cur_entry.distance) {
            cur_entry.id = static_cast<PID>(i);
            cur_entry.distance = distance;
        }
    }

    PID nearest_neighbor = 0;
    float min_dist = FLT_MAX;
    for (auto& candi : best_entries) {
        if (candi.distance < min_dist) {
            nearest_neighbor = candi.id;
            min_dist = candi.distance;
        }
    }
    return nearest_neighbor;
}

inline auto compute_centroid(
    const float* data, size_t num_points, size_t dim, size_t num_threads
) {
    std::vector<std::vector<double>> all_results(num_threads, std::vector<double>(dim, 0));

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        std::vector<double>& cur_results = all_results[tid];
        const float* cur_data = data + (dim * i);
        for (size_t k = 0; k < dim; ++k) {
            cur_results[k] += cur_data[k];
        }
    }

    std::vector<double> centroid_d(dim, 0);
    std::vector<float> centroid(dim, 0);
    for (auto& one_res : all_results) {
        for (size_t i = 0; i < dim; ++i) {
            centroid_d[i] += one_res[i];
        }
    }
    double inv_num_points = 1.0 / static_cast<double>(num_points);
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = static_cast<float>(centroid_d[i] * inv_num_points);
    }

    return centroid;
}

template <typename T>
inline void matrix_padding(
    const RowMatrix<T>& mat, RowMatrix<T>& padded_mat, size_t padded_dim
) {
    size_t num_rows = mat.rows();
    size_t dim = mat.cols();

    padded_mat = RowMatrix<T>(num_rows, padded_dim);
    padded_mat.setZero();

    for (size_t i = 0; i < num_rows; ++i) {
        std::copy(&mat(i, 0), &mat(i, 0) + dim, &padded_mat(i, 0));
    }
}
}  // namespace symqg::space