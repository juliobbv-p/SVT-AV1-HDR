/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

/*!\file
  * \brief Describes film grain parameters and film grain synthesis
  *
  */
#ifndef AOM_AOM_GRAIN_SYNTHESIS_H_
#define AOM_AOM_GRAIN_SYNTHESIS_H_

#include <stdint.h>
#include "EbSvtAv1.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline bool svt_aom_film_grain_params_valid(const AomFilmGrain* grain, bool is_yuv420) {
    if (grain->apply_grain == 0) {
        return true;
    }
    if (grain->apply_grain != 1 || (grain->bit_depth != 8 && grain->bit_depth != 10) ||
        (uint32_t)grain->update_parameters > 1 || (uint32_t)grain->ignore_ref > 1 ||
        (uint32_t)grain->chroma_scaling_from_luma > 1 || (uint32_t)grain->overlap_flag > 1 ||
        (uint32_t)grain->clip_to_restricted_range > 1 || grain->scaling_shift < 8 || grain->scaling_shift > 11 ||
        (uint32_t)grain->ar_coeff_lag > 3 || grain->ar_coeff_shift < 6 || grain->ar_coeff_shift > 9 ||
        (uint32_t)grain->grain_scale_shift > 3 || (uint32_t)grain->num_y_points > 14 ||
        (uint32_t)grain->num_cb_points > 10 || (uint32_t)grain->num_cr_points > 10 ||
        (is_yuv420 && !grain->chroma_scaling_from_luma && grain->num_y_points &&
         (!grain->num_cb_points != !grain->num_cr_points))) {
        return false;
    }
    const int counts[]           = {grain->num_y_points, grain->num_cb_points, grain->num_cr_points};
    const int32_t (*points[])[2] = {grain->scaling_points_y, grain->scaling_points_cb, grain->scaling_points_cr};
    const int32_t* coeffs[]      = {grain->ar_coeffs_y, grain->ar_coeffs_cb, grain->ar_coeffs_cr};
    for (int plane = 0; plane < 3; ++plane) {
        for (int i = 0; i < counts[plane]; ++i) {
            if ((uint32_t)points[plane][i][0] > 255 || (uint32_t)points[plane][i][1] > 255 ||
                (i && points[plane][i][0] <= points[plane][i - 1][0])) {
                return false;
            }
        }
        const int n = 2 * grain->ar_coeff_lag * (grain->ar_coeff_lag + 1) + (plane && grain->num_y_points > 0);
        if (counts[plane] || (plane && grain->chroma_scaling_from_luma)) {
            for (int i = 0; i < n; ++i) {
                if (coeffs[plane][i] < -128 || coeffs[plane][i] > 127) {
                    return false;
                }
            }
        }
    }
    if ((grain->num_cb_points &&
         ((uint32_t)grain->cb_mult > 255 || (uint32_t)grain->cb_luma_mult > 255 || (uint32_t)grain->cb_offset > 511)) ||
        (grain->num_cr_points &&
         ((uint32_t)grain->cr_mult > 255 || (uint32_t)grain->cr_luma_mult > 255 || (uint32_t)grain->cr_offset > 511))) {
        return false;
    }
    return true;
}

int32_t svt_aom_film_grain_params_equal(const AomFilmGrain* pars_a, const AomFilmGrain* pars_b);

/*!\brief Add film grain
     *
     * Add film grain to an image
     *
     * \param[in]    grain_params     Grain parameters
     * \param[in]    luma             luma plane
     * \param[in]    cb               cb plane
     * \param[in]    cr               cr plane
     * \param[in]    height           luma plane height
     * \param[in]    width            luma plane width
     * \param[in]    luma_stride      luma plane stride
     * \param[in]    chroma_stride    chroma plane stride
     */
void svt_av1_add_film_grain_run(const AomFilmGrain* grain_params, uint8_t* luma, uint8_t* cb, uint8_t* cr,
                                int32_t height, int32_t width, int32_t luma_stride, int32_t chroma_stride,
                                int32_t use_high_bit_depth, int32_t chroma_subsamp_y, int32_t chroma_subsamp_x);

void svt_aom_fgn_copy_rect(uint8_t* src, int32_t src_stride, uint8_t* dst, int32_t dst_stride, int32_t width,
                           int32_t height, int32_t use_high_bit_depth);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOM_AOM_GRAIN_SYNTHESIS_H_
