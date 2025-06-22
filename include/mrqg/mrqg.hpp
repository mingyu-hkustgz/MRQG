//
// Created by bld on 25-6-13.
//
#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

#include "mrqg_builder.hpp"
#include "mrqg_query.hpp"
#include "mrqg_scanner.hpp"

#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#ifndef MRQG_MRQG_HPP
#define MRQG_MRQG_HPP


namespace symqg {

    class ResidualQuantizedGraph {
    private:
        size_t num_points_ = 0;    // num points
        size_t degree_bound_ = 0;  // degree bound
        size_t dimension_ = 0;     // dimension
        size_t padded_dim_ = 0;    // padded dimension
        size_t flop_dim_ = 0;      // First stage dimension
        size_t turn_dim_ = 0;      // Second stage dimension
        PID entry_point_ = 0;      // Entry point of graph
        symqg::data::Array<
                float,
                std::vector<size_t>,
                symqg::memory::AlignedAllocator<
                        float,
                        1 << 22,
                        true>>
                data_;  // vectors + graph + quantization codes
        symqg::QGScanner scanner_;
        symqg::FHTRotator rotator_;
        symqg::HashBasedBooleanSet visited_;
        symqg::buffer::SearchBuffer search_pool_;

        /*
         * Position of different data in each row
         *      RawData + QuantizationCodes + Factors + neighborIDs
         * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
         * we do not need to store the degree for each vertex
         */
        size_t code_offset_ = 0;      // pos of packed code
        size_t factor_offset_ = 0;    // pos of Factor
        size_t neighbor_offset_ = 0;  // pos of Neighbors
        size_t row_offset_ = 0;       // length of entire row

        void initialize();

        // search on quantized graph
        void search_qg(
                const float *__restrict__ query, uint32_t knn, uint32_t *__restrict__ results
        );

        void copy_vectors(const float *);

        [[nodiscard]] float *get_vector(PID data_id) {
            return &data_.at(row_offset_ * data_id);
        }

        [[nodiscard]] const float *get_vector(PID data_id) const {
            return &data_.at(row_offset_ * data_id);
        }

        [[nodiscard]] uint8_t *get_packed_code(PID data_id) {
            return reinterpret_cast<uint8_t *>(&data_.at((row_offset_ * data_id) + code_offset_)
            );
        }

        [[nodiscard]] const uint8_t *get_packed_code(PID data_id) const {
            return reinterpret_cast<const uint8_t *>(
                    &data_.at((row_offset_ * data_id) + code_offset_)
            );
        }

        [[nodiscard]] float *get_factor(PID data_id) {
            return &data_.at((row_offset_ * data_id) + factor_offset_);
        }

        [[nodiscard]] const float *get_factor(PID data_id) const {
            return &data_.at((row_offset_ * data_id) + factor_offset_);
        }

        [[nodiscard]] PID *get_neighbors(PID data_id) {
            return reinterpret_cast<PID *>(&data_.at((row_offset_ * data_id) + neighbor_offset_)
            );
        }

        [[nodiscard]] const PID *get_neighbors(PID data_id) const {
            return reinterpret_cast<const PID *>(
                    &data_.at((row_offset_ * data_id) + neighbor_offset_)
            );
        }

        void
        find_candidates(PID, size_t, std::vector<Candidate<float>> &, HashBasedBooleanSet &,
                        const std::vector<uint32_t> &)
        const;

        void update_qg(PID, const std::vector<Candidate<float>> &);

        void update_results(buffer::ResultBuffer &, const float *);

        float scan_neighbors(
                const QGQuery &q_obj,
                const float *cur_data,
                float *appro_dist,
                buffer::SearchBuffer &search_pool,
                uint32_t cur_degree
        ) const;

    public:
        explicit ResidualQuantizedGraph(size_t, size_t, size_t, size_t, size_t);

        [[nodiscard]] auto num_vertices() const { return this->num_points_; }

        [[nodiscard]] auto dimension() const { return this->dimension_; }

        [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

        [[nodiscard]] auto entry_point() const { return this->entry_point_; }

        void set_ep(PID entry) { this->entry_point_ = entry; };

        void save_index(const char *) const;

        void load_index(const char *);

        void set_ef(size_t);

        /* search and copy results to KNN */
        void search(
                const float *__restrict__ query, uint32_t knn, uint32_t *__restrict__ results
        );

    };

    inline ResidualQuantizedGraph::ResidualQuantizedGraph(size_t num, size_t max_deg, size_t dim, size_t f_dim,
                                                          size_t t_dim)
            : num_points_(num),
            degree_bound_(max_deg),
            dimension_(dim),
            flop_dim_(dim / 2),
            turn_dim_(dim / 4 * 3),
            scanner_(dim, degree_bound_),
            rotator_(dimension_),
            visited_(100),
            search_pool_(0) {
        initialize();
    }

    inline void ResidualQuantizedGraph::initialize() {
        /* check size */

        this->code_offset_ = dimension_;  // Pos of packed code (aligned)
        this->factor_offset_ =
                code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor
        this->neighbor_offset_ =
                factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
        this->row_offset_ = neighbor_offset_ + degree_bound_;

        /* Allocate memory of data*/
        data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
                std::vector<size_t>{num_points_, row_offset_}
        );
    }

    inline void ResidualQuantizedGraph::copy_vectors(const float* data) {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_points_; ++i) {
            const float* src = data + (dimension_ * i);
            float* dst = get_vector(i);
            std::copy(src, src + dimension_, dst);
        }
        std::cout << "\tVectors Copied\n";
    }

    inline void ResidualQuantizedGraph::save_index(const char* filename) const {
        std::cout << "Saving quantized graph to " << filename << '\n';
        std::ofstream output(filename, std::ios::binary);
        assert(output.is_open());

        /* Basic variants */
        output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

        /* Data */
        data_.save(output);

        /* Rotator */
        this->rotator_.save(output);

        output.close();
        std::cout << "\tQuantized graph saved!\n";
    }

    inline void ResidualQuantizedGraph::load_index(const char* filename) {
        std::cout << "loading quantized graph " << filename << '\n';

        /* Check existence */
        if (!file_exists(filename)) {
            std::cerr << "Index does not exist!\n";
            abort();
        }

        /* Check file size */
        size_t filesize = get_filesize(filename);
        size_t correct_size = sizeof(PID) + (sizeof(float) * num_points_ * row_offset_) +
                              (sizeof(float) * padded_dim_);
        if (filesize != correct_size) {
            std::cerr << "Index file size error! Please make sure the index and "
                         "init parameters are correct\n";
            abort();
        }

        std::ifstream input(filename, std::ios::binary);
        assert(input.is_open());

        /* Basic variants */
        input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

        /* Data */
        data_.load(input);

        /* Rotator */
        this->rotator_.load(input);

        input.close();
        std::cout << "Quantized graph loaded!\n";
    }

    inline void ResidualQuantizedGraph::set_ef(size_t cur_ef) {
        this->search_pool_.resize(cur_ef);
        this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
    }

    /*
     * search single query
     */
    inline void ResidualQuantizedGraph::search(
            const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    ) {
        /* Init query matrix */
        this->visited_.clear();
        this->search_pool_.clear();
        search_qg(query, knn, results);
    }
    /**
     * @brief search on qg
     *
     * @param query     unrotated query vector, dimension_ elements
     * @param knn       num of nearest neighbors
     * @param results   searh res
     */
    inline void ResidualQuantizedGraph::search_qg(
            const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    ) {
        // query preparation
        QGQuery q_obj(query, padded_dim_);
        q_obj.query_prepare(rotator_, scanner_);

        /* Searching pool initialization */
        search_pool_.insert(this->entry_point_, FLT_MAX);

        /* Result pool */
        buffer::ResultBuffer res_pool(knn);

        /* Current version of fast scan compute 32 distances */
        std::vector<float> appro_dist(degree_bound_);  // approximate dis

        while (search_pool_.has_next()) {
            PID cur_node = search_pool_.pop();
            if (visited_.get(cur_node)) {
                continue;
            }
            visited_.set(cur_node);

            float sqr_y = scan_neighbors(
                    q_obj,
                    get_vector(cur_node),
                    appro_dist.data(),
                    this->search_pool_,
                    this->degree_bound_
            );
            res_pool.insert(cur_node, sqr_y);
        }

        update_results(res_pool, query);
        res_pool.copy_results(results);
    }


    // scan a data row (including data vec and quantization codes for its neighbors)
// return exact distnace for current vertex
    inline float ResidualQuantizedGraph::scan_neighbors(
            const QGQuery& q_obj,
            const float* cur_data,
            float* appro_dist,
            buffer::SearchBuffer& search_pool,
            uint32_t cur_degree
    ) const {
        float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

        /* Compute approximate distance by Fast Scan */
        const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
        const auto* factor = &cur_data[factor_offset_];
        this->scanner_.scan_neighbors(
                appro_dist,
                q_obj.lut().data(),
                sqr_y,
                q_obj.lower_val(),
                q_obj.width(),
                q_obj.sumq(),
                packed_code,
                factor
        );

        const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
        for (uint32_t i = 0; i < cur_degree; ++i) {
            PID cur_neighbor = ptr_nb[i];
            float tmp_dist = appro_dist[i];
#if defined(DEBUG)
            std::cout << "Neighbor ID " << cur_neighbor << '\n';
        std::cout << "Appro " << appro_dist[i] << '\t';
        float __gt_dist__ = l2_sqr(query, get_vector(cur_neighbor), dimension_);
        std::cout << "GT " << __gt_dist__ << '\t';
        std::cout << "Error " << (appro_dist[i] - __gt_dist__) / __gt_dist__ << '\t';
        std::cout << "sqr_y " << sqr_y << '\n';
#endif
            if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
                continue;
            }
            search_pool.insert(cur_neighbor, tmp_dist);
            memory::mem_prefetch_l2(
                    reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
            );
        }

        return sqr_y;
    }
}


#endif //MRQG_MRQG_HPP
