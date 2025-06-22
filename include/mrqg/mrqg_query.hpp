#pragma once

#include <cstdint>

#include "../common.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "../space/l2.hpp"
#include "../utils/scalar_quantize.hpp"
#include "./mrqg_scanner.hpp"

namespace symqg {
    class MRQGQuery {
    private:
        const float* query_data_ = nullptr;
        const float* res_data_ = nullptr;
        std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_;
        size_t flop_dim_ = 0;
        size_t res_dim_ = 0;
        float width_ = 0;
        float lower_val_ = 0;
        float upper_val_ = 0;
        float res_norm_ = 0;
        int32_t sumq_ = 0;

    public:
        explicit MRQGQuery(const float* q,const float* r, size_t flop_dim, size_t dim)
                : query_data_(q)
                , res_data_(r)
                , lut_(flop_dim << 2)
                , flop_dim_(flop_dim) // no pad dim
                , res_dim_(dim - flop_dim)
                {}

        void query_prepare(const FHTRotator& rotator, const MRQGScanner& scanner) {
            // rotate query
            std::vector<float, memory::AlignedAllocator<float>> rd_query(flop_dim_);
            rotator.rotate(query_data_, rd_query.data());

            // quantize query
            std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(flop_dim_);
            scalar::data_range(rd_query.data(), flop_dim_, lower_val_, upper_val_);
            width_ = (upper_val_ - lower_val_) / ((1 << QG_BQUERY) - 1);
            scalar::quantize(
                    byte_query.data(), rd_query.data(), flop_dim_, lower_val_, width_, sumq_
            );

            // pack lut
            scanner.pack_lut(byte_query.data(), lut_.data());
            res_norm_ = space::l2_sqr_single(res_data_, res_dim_);
        }

        [[nodiscard]] const float& width() const { return width_; }

        [[nodiscard]] const float& lower_val() const { return lower_val_; }

        [[nodiscard]] const int32_t& sumq() const { return sumq_; }

        [[nodiscard]] const float& res_norm() const { return res_norm_; }

        [[nodiscard]] const std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>>& lut(
        ) const {
            return lut_;
        }

        [[nodiscard]] const float* query_data() const { return query_data_; }
        [[nodiscard]] const float* res_data() const { return res_data_; }
    };
}  // namespace symqg