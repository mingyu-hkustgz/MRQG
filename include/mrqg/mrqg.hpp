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

#include "mrqg_query.hpp"
#include "mrqg_scanner.hpp"

#include "../common.hpp"
#include "../quantization/rabitq.hpp"
#include "../space/l2.hpp"
#include "../space/inner_product.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"


namespace symqg {
    /**
     * @brief this Factor only for illustration, the true storage is continous
     * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
     *
     */
    struct Factor {
        float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0 + res vec norm
        float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
        float factor_vq;  // Factor of v_l * ||q_r||
    };

    class ResidualQuantizedGraph {
        friend class MRQGBuilder;

    private:
        size_t num_points_ = 0;    // num points
        size_t degree_bound_ = 0;  // degree bound
        size_t dimension_ = 0;     // dimension
        size_t flop_dim_ = 0;      // First stage dimension
        size_t res_dim_ = 0;       // Residual stage dimension
        PID entry_point_ = 0;      // Entry point of graph
        symqg::data::Array<
                float,
                std::vector<size_t>,
                symqg::memory::AlignedAllocator<
                        float,
                        1 << 22,
                        true>>
                data_;  // vectors + graph + quantization codes
        symqg::MRQGScanner scanner_;
        symqg::FHTRotator rotator_;
        symqg::HashBasedBooleanSet visited_;
        symqg::buffer::SearchBuffer search_pool_;

        /*
         * Position of different data in each row
         *      RawData + QuantizationCodes + Factors + neighborIDs
         * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
         * we do not need to store the degree for each vertex
         */
        size_t vec_offset_ = 0;
        size_t code_offset_ = 0;      // pos of packed code
        size_t factor_offset_ = 0;    // pos of Factor
        size_t neighbor_offset_ = 0;  // pos of Neighbors
        size_t residual_offset_ = 0;  // pos of Residual Dim
        size_t row_offset_ = 0;       // length of entire row

        void initialize();

        // search on quantized graph
        void search_qg(const float *__restrict__ query, uint32_t knn, uint32_t *__restrict__ results);

        void copy_vectors(const float *);

        void copy_norm_vectors(const float *);

        [[nodiscard]] float *get_vector_norm(PID data_id) {
            return &data_.at(row_offset_ * data_id);
        }

        [[nodiscard]] const float *get_vector_norm(PID data_id) const{
            return &data_.at(row_offset_ * data_id);
        }

        [[nodiscard]] float *get_res_vector(PID data_id)  {
            return &data_.at(row_offset_ * data_id + residual_offset_);
        }

        [[nodiscard]] float *get_vector(PID data_id) {
            return &data_.at(row_offset_ * data_id + vec_offset_);
        }

        [[nodiscard]] const float *get_vector(PID data_id) const {
            return &data_.at(row_offset_ * data_id + vec_offset_);
        }

        [[nodiscard]] const float *get_res_vector(PID data_id) const {
            return &data_.at(row_offset_ * data_id + residual_offset_);
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

        void update_results(buffer::ResultBuffer &, const float *, float);

        float scan_neighbors(
                const MRQGQuery &q_obj,
                const float *cur_data,
                float *appro_dist,
                buffer::SearchBuffer &search_pool,
                uint32_t cur_degree
        ) const;

    public:
        explicit ResidualQuantizedGraph(size_t, size_t, size_t, size_t);

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

    inline ResidualQuantizedGraph::ResidualQuantizedGraph(size_t num, size_t max_deg, size_t dim, size_t f_dim)
            : num_points_(num),
              degree_bound_(max_deg),
              dimension_(dim),
              flop_dim_(f_dim),
              res_dim_(dim - flop_dim_),
              scanner_(f_dim, degree_bound_),
              rotator_(f_dim),
              visited_(100),
              search_pool_(0) {
        initialize();
    }

    inline void ResidualQuantizedGraph::initialize() {
        /* check size */
        assert(flop_dim_ % 64 == 0);

        this->vec_offset_ = 1;           // Pos of vec residual norm (aligned)
        this->code_offset_ = vec_offset_ + flop_dim_;  // Pos of packed code (aligned)
        this->factor_offset_ = code_offset_ + flop_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor
        this->neighbor_offset_ =
                factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
        this->residual_offset_ = neighbor_offset_ + degree_bound_;
        this->row_offset_ = residual_offset_ + res_dim_;

        /* Allocate memory of data*/
        data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
                std::vector<size_t>{num_points_, row_offset_}
        );
    }


    inline void ResidualQuantizedGraph::copy_vectors(const float *data) {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_points_; ++i) {
            const float *src = data + (dimension_ * i);
            float *dst = get_vector(i);
            std::copy(src, src + dimension_, dst);
        }
        std::cout << "\tVectors Copied\n";
    }

    inline void ResidualQuantizedGraph::copy_norm_vectors(const float *data) {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_points_; ++i) {
            const float *src = data + (dimension_ * i);
            float res_norm = space::l2_sqr_single(src + flop_dim_, res_dim_);
            *get_vector_norm(i) = res_norm;
            float *dst = get_vector(i);
            std::copy(src, src + flop_dim_, dst);
            float *nst = get_res_vector(i);
            std::copy(src + flop_dim_, src + dimension_, nst);
        }
        std::cout << "\tVectors Copied\n";
    }

    inline void ResidualQuantizedGraph::save_index(const char *filename) const {
        std::cout << "Saving quantized graph to " << filename << '\n';
        std::ofstream output(filename, std::ios::binary);
        assert(output.is_open());

        /* Basic variants */
        output.write(reinterpret_cast<const char *>(&entry_point_), sizeof(PID));

        /* Data */
        data_.save(output);

        /* Rotator */
        this->rotator_.save(output);

        output.close();
        std::cout << "\tQuantized graph saved!\n";
    }

    inline void ResidualQuantizedGraph::load_index(const char *filename) {
        std::cout << "loading quantized graph " << filename << '\n';

        /* Check existence */
        if (!file_exists(filename)) {
            std::cerr << "Index does not exist!\n";
            abort();
        }

        /* Check file size */
        size_t filesize = get_filesize(filename);
        size_t correct_size = sizeof(PID) + (sizeof(float) * num_points_ * row_offset_) +
                              (sizeof(float) * dimension_);
        if (filesize != correct_size) {
            std::cerr << "Index file size error! Please make sure the index and "
                         "init parameters are correct\n";
            abort();
        }

        std::ifstream input(filename, std::ios::binary);
        assert(input.is_open());

        /* Basic variants */
        input.read(reinterpret_cast<char *>(&entry_point_), sizeof(PID));

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
            const float *__restrict__ query, uint32_t knn, uint32_t *__restrict__ results
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
            const float *__restrict__ query, uint32_t knn, uint32_t *__restrict__ results
    ) {
        // query preparation
        MRQGQuery q_obj(query,query+flop_dim_, flop_dim_, dimension_);
        q_obj.query_prepare(rotator_, scanner_);

        /* Searching pool initialization */
        search_pool_.insert(this->entry_point_, FLT_MAX);

        /* Result pool */
        buffer::ResultBuffer res_pool(knn);

        /* Current version of fast scan compute 32 distances */
        std::vector<float> appro_dist(degree_bound_);  // approximate dis

        while (search_pool_.has_next()) {
            auto cur_pos = search_pool_.get_pos();
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
            if(cur_pos < search_pool_.get_size()>>1)
                sqr_y -= 2.0F * space::ip_sim(q_obj.res_data(), get_res_vector(cur_node), res_dim_);
            res_pool.insert(cur_node, sqr_y);
        }

        update_results(res_pool, query, q_obj.res_norm());

        res_pool.copy_results(results);
    }


    // scan a data row (including data vec and quantization codes for its neighbors)
// return exact distnace for current vertex
    inline float ResidualQuantizedGraph::scan_neighbors(
            const MRQGQuery &q_obj,
            const float *cur_data,
            float *appro_dist,
            buffer::SearchBuffer &search_pool,
            uint32_t cur_degree
    ) const {
        float sqr_y = cur_data[0] + space::l2_sqr(q_obj.query_data(), cur_data + 1, flop_dim_) + q_obj.res_norm();

        /* Compute approximate distance by Fast Scan */
        const auto *packed_code = reinterpret_cast<const uint8_t *>(&cur_data[code_offset_]);
        const auto *factor = &cur_data[factor_offset_];
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

        const PID *ptr_nb = reinterpret_cast<const PID *>(&cur_data[neighbor_offset_]);
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
                    reinterpret_cast<const char *>(get_vector(search_pool.next_id())), 10
            );
        }
        return sqr_y;
    }

    inline void ResidualQuantizedGraph::update_results(
            buffer::ResultBuffer &result_pool, const float *query, float res_norm
    ) {
        if (result_pool.is_full()) {
            return;
        }

        auto ids = result_pool.ids();
        for (PID data_id: ids) {
            PID *ptr_nb = get_neighbors(data_id);
            for (uint32_t i = 0; i < this->degree_bound_; ++i) {
                PID cur_neighbor = ptr_nb[i];
                if (!visited_.get(cur_neighbor)) {
                    visited_.set(cur_neighbor);
                    float *neighbor_data = get_vector_norm(cur_neighbor);
                    result_pool.insert(
                            cur_neighbor, res_norm + neighbor_data[0] +  space::l2_sqr(query, neighbor_data + 1, flop_dim_)
                    );
                }
            }
            if (result_pool.is_full()) {
                break;
            }
        }
    }

    // find candidate neighbors for cur_id, exclude the vertex itself
    inline void ResidualQuantizedGraph::find_candidates(
            PID cur_id,
            size_t search_ef,
            std::vector<Candidate<float>> &results,
            HashBasedBooleanSet &vis,
            const std::vector<uint32_t> &degrees
    ) const {
        const float *query = get_vector(cur_id);
        const float *res = get_res_vector(cur_id);
        MRQGQuery q_obj(query, res, flop_dim_, dimension_);
        q_obj.query_prepare(rotator_, scanner_);

        /* Searching pool initialization */
        buffer::SearchBuffer tmp_pool(search_ef);
        tmp_pool.insert(this->entry_point_, 1e10);
        memory::mem_prefetch_l1(
                reinterpret_cast<const char *>(get_vector(this->entry_point_)), 10
        );

        /* Current version of fast scan compute 32 distances */
        std::vector<float> appro_dist(degree_bound_);  // approximate dis
        while (tmp_pool.has_next()) {
            auto cur_pos = tmp_pool.get_pos();
            auto cur_candi = tmp_pool.pop();
            if (vis.get(cur_candi)) {
                continue;
            }
            vis.set(cur_candi);
            auto cur_degree = degrees[cur_candi];
            auto sqr_y = scan_neighbors(
                    q_obj, get_vector_norm(cur_candi), appro_dist.data(), tmp_pool, cur_degree);
            if (cur_candi != cur_id) {
                if(cur_pos < search_ef>>2)
                    sqr_y -= 2.0F * space::ip_sim(q_obj.res_data(), get_res_vector(cur_candi), res_dim_);
                results.emplace_back(cur_candi, sqr_y);
            }
        }
    }

    inline void ResidualQuantizedGraph::update_qg(
            PID cur_id, const std::vector<Candidate<float>> &new_neighbors
    ) {
        size_t cur_degree = new_neighbors.size();

        if (cur_degree == 0) {
            return;
        }
        // copy neighbors
        PID *neighbor_ptr = get_neighbors(cur_id);
        for (size_t i = 0; i < cur_degree; ++i) {
            neighbor_ptr[i] = new_neighbors[i].id;
        }

        RowMatrix<float> x_pad(cur_degree, flop_dim_);  // pca proj neighbors mat
        RowMatrix<float> c_pad(1, flop_dim_);           // pca proj duplicate centroid mat
        x_pad.setZero();
        c_pad.setZero();

        /* Copy data */
        for (size_t i = 0; i < cur_degree; ++i) {
            auto neighbor_id = new_neighbors[i].id;
            const auto* cur_data = get_vector(neighbor_id);
            std::copy(cur_data, cur_data + flop_dim_, &x_pad(static_cast<long>(i), 0));
        }
        const auto* cur_cent = get_vector(cur_id);
        std::copy(cur_cent, cur_cent + flop_dim_, &c_pad(0, 0));

        /* rotate Matrix */
        RowMatrix<float> x_rotated(cur_degree, flop_dim_);
        RowMatrix<float> c_rotated(1, flop_dim_);
        for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
            this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
        }
        this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

        // Get codes and factors for rabitq
        float* fac_ptr = get_factor(cur_id);
        float* triple_x = fac_ptr;
        float* factor_dq = triple_x + this->degree_bound_;
        float* factor_vq = factor_dq + this->degree_bound_;
        rabitq_codes(
                x_rotated, c_rotated, get_packed_code(cur_id), triple_x, factor_dq, factor_vq
        );
    }

}

