#pragma once

// GQA-packed FlashAttention decode kernel for quantized K/V.
//
// Motivation:
//   The generic vector kernel (fattn-vec.cuh) assigns one CTA per (query head, query tile), so for GQA
//   with a ratio of 6 the K/V cache is read 6x and V is dequantized 6x per attention op. The MMA/TILE
//   kernels do pack GQA heads (ncols2), but they require the entire K/V cache to be converted to F16 in
//   VRAM first (see launch_fattn), which costs O(context) bandwidth per token step.
//
//   This kernel packs all GQA_PACK query heads that share a KV head into a single CTA and reads the
//   quantized K/V once. K/V rows are dequantized to FP32 on the fly (no F16 staging buffer) and the
//   softmax numerator is accumulated in registers.
//
//   Work distribution: the (query position, head) pairs handled by a CTA are split between the warps
//   (cpw columns per warp). Every warp iterates over *all* KV rows of the CTA's range, because a column
//   needs the contributions of all rows to compute its softmax. The K/V rows are therefore loaded (and
//   dequantized) redundantly by each warp, but those loads hit the L1 cache.
//
//   Restrictions (checked by the dispatcher):
//     * D == 256 (D % (4*WARP_SIZE) == 0 and D/WARP_SIZE elements per lane)
//     * query batch == 2 (the MTP verification batch on the target model)
//     * ncols1*GQA_PACK % nwarps == 0
//     * type_K == type_V == GGML_TYPE_Q4_0
//     * no logit softcap, no sinks
//
//   Output contract is the same as for the vector kernel: without KV splitting the normalized result is
//   written to dst; with KV splitting (gridDim.y > 1) unnormalized partials are written to dst and
//   (KQ_max, KQ_sum) to dst_meta so that flash_attn_combine_results<DV> can merge them.

#include "common.cuh"
#include "fattn-common.cuh"

template<int D, int GQA_PACK, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(128, 1)
static __global__ void flash_attn_ext_vec_gqa(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const char * __restrict__ sinks_ptr,
        const int  * __restrict__ KV_max_ptr,
        float      * __restrict__ dst_ptr,
        float2     * __restrict__ dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    GGML_UNUSED(ne00);
    GGML_UNUSED(ne03);
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(ne31);
    GGML_UNUSED(ne32);
    GGML_UNUSED(nb32);

#ifdef FLASH_ATTN_AVAILABLE
    ggml_cuda_pdl_lc();

    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    GGML_UNUSED(sinks);

    constexpr int nthreads = 128;
    constexpr int nwarps   = nthreads/WARP_SIZE;
    constexpr int ncols1   = 2;                      // query positions per CTA
    constexpr int ncols    = ncols1*GQA_PACK;        // (query position, head) pairs handled by the CTA
    constexpr int cpw      = ncols/nwarps;           // pairs per warp
    constexpr int DL       = D/WARP_SIZE;            // elements per lane and pair
    constexpr int NCALLS   = DL/4;                   // dequantize calls per lane and row

    static_assert(D % (4*WARP_SIZE) == 0, "D must be a multiple of 128");
    static_assert(ncols % nwarps == 0, "ncols must be a multiple of the number of warps");

    constexpr dequantize_V_t dequantize_K = get_dequantize_V<type_K, float, 4>();
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, 4>();

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    const int nheads_gqa = ne02 / GQA_PACK;
    const int sequence   = blockIdx.z / nheads_gqa;
    const int head0      = (blockIdx.z - sequence*nheads_gqa) * GQA_PACK;
    const int gqa_ratio  = ne02 / ne12;

    const int ic0 = blockIdx.x * ncols1;

    const char * Qb = Q + nb03*sequence + nb02*head0            + nb01*ic0;
    const char * Kb = K + nb13*sequence + nb12*(head0/gqa_ratio);
    const char * Vb = V + nb23*sequence + nb22*(head0/gqa_ratio);

    const half * maskh = (const half *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    // Column index mapping: column jc == cq*GQA_PACK + ch, i.e. query position ic0 + cq for head head0 + ch.
    // Warp w owns columns [w*cpw, (w + 1)*cpw).
    const int j0 = warp*cpw;

    // Load this lane's slice of Q (DL contiguous elements per column) and pre-scale it:
    ggml_cuda_pdl_sync();
    float Q_reg[cpw][DL];
#pragma unroll
    for (int j = 0; j < cpw; ++j) {
        const int jc = j0 + j;
        const float4 * Q4 = (const float4 *) (Qb + (jc % GQA_PACK)*nb02 + (jc / GQA_PACK)*nb01);

        const float4 qa = Q4[2*lane + 0];
        const float4 qb = Q4[2*lane + 1];

        Q_reg[j][0] = scale*qa.x;
        Q_reg[j][1] = scale*qa.y;
        Q_reg[j][2] = scale*qa.z;
        Q_reg[j][3] = scale*qa.w;
        Q_reg[j][4] = scale*qb.x;
        Q_reg[j][5] = scale*qb.y;
        Q_reg[j][6] = scale*qb.z;
        Q_reg[j][7] = scale*qb.w;
    }

    float KQ_max[cpw];
    float KQ_sum[cpw];
    float VKQ[cpw][DL];
#pragma unroll
    for (int j = 0; j < cpw; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
#pragma unroll
        for (int i = 0; i < DL; ++i) {
            VKQ[j][i] = 0.0f;
        }
    }

    float slope[cpw];
#pragma unroll
    for (int j = 0; j < cpw; ++j) {
        slope[j] = get_alibi_slope(max_bias, head0 + (j0 + j) % GQA_PACK, n_head_log2, m0, m1);
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    Kb += blockIdx.y*nthreads*nb11;
    Vb += blockIdx.y*nthreads*nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             Kb += gridDim.y*nthreads*nb11, Vb += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {
#pragma unroll 4
        for (int ir = 0; ir < nthreads; ++ir) {
            if (k_VKQ_0 + ir >= k_VKQ_max) {
                break;
            }
            const char * Kr = Kb + ir*nb11;
            const char * Vr = Vb + ir*nb21;

            // Dequantize this lane's slice of K and V (DL contiguous elements each):
            float K_reg[DL];
            float V_reg[DL];
#pragma unroll
            for (int c = 0; c < NCALLS; ++c) {
                if constexpr (type_K == GGML_TYPE_Q4_0) {
                    dequantize_V_q4_0<float, 4>(Kr, K_reg + 4*c, DL*lane + 4*c);
                } else {
                    dequantize_K(Kr, K_reg + 4*c, DL*lane + 4*c);
                }
                if constexpr (type_V == GGML_TYPE_Q4_0) {
                    dequantize_V_q4_0<float, 4>(Vr, V_reg + 4*c, DL*lane + 4*c);
                } else {
                    dequantize_V(Vr, V_reg + 4*c, DL*lane + 4*c);
                }
            }

#pragma unroll
            for (int j = 0; j < cpw; ++j) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    dot += Q_reg[j][i]*K_reg[i];
                }
                dot = warp_reduce_sum(dot);

                if (use_logit_softcap) {
                    dot = logit_softcap*tanhf(dot);
                }

                if (mask) {
                    dot += slope[j]*__half2float(maskh[((j0 + j)/GQA_PACK)*ne11 + ir]);
                }

                // Online softmax. FATTN_KQ_MAX_OFFSET is added to the max. like in the vector kernel to
                // increase the numerical range that the accumulators can represent.
                const float dot_off = dot + FATTN_KQ_MAX_OFFSET;
                if (dot_off > KQ_max[j]) {
                    const float ms = expf(KQ_max[j] - dot_off);
                    KQ_max[j] = dot_off;
                    KQ_sum[j] = KQ_sum[j]*ms + 1.0f;
#pragma unroll
                    for (int i = 0; i < DL; ++i) {
                        VKQ[j][i] = VKQ[j][i]*ms + V_reg[i];
                    }
                } else {
                    const float p = expf(dot_off - KQ_max[j]);
                    KQ_sum[j] += p;
#pragma unroll
                    for (int i = 0; i < DL; ++i) {
                        VKQ[j][i] = fmaf(p, V_reg[i], VKQ[j][i]);
                    }
                }
            }
        }
    }

    // Write the output. Every column is fully accumulated by its owning warp, no cross-warp reduction is
    // needed. Each lane holds the elements [DL*lane, DL*(lane + 1)) of every column.
#pragma unroll
    for (int j = 0; j < cpw; ++j) {
        const int jc = j0 + j;
        const int cq = jc / GQA_PACK;
        const int ch = jc % GQA_PACK;

        const float inv_kqsum = gridDim.y == 1 ? 1.0f/KQ_sum[j] : 1.0f;
        const size_t i_dst = (((sequence*int(ne01.z) + ic0 + cq)*ne02 + head0 + ch)*gridDim.y + blockIdx.y)*D;

#pragma unroll
        for (int i = 0; i < DL; ++i) {
            dst[i_dst + DL*lane + i] = VKQ[j][i]*inv_kqsum;
        }

        if (gridDim.y != 1 && lane == 0) {
            dst_meta[((sequence*int(ne01.z) + ic0 + cq)*ne02 + head0 + ch)*gridDim.y + blockIdx.y] =
                make_float2(KQ_max[j], KQ_sum[j]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int D, int GQA_PACK, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_gqa_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int nthreads = 128;
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec_gqa<D, GQA_PACK, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = false;
    const bool need_f16_V = false;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, 2, GQA_PACK>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, int GQA_PACK, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_gqa_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    GGML_ASSERT(Q->ne[1] == 2);
    GGML_ASSERT(logit_softcap == 0.0f);
    GGML_ASSERT(dst->src[4] == nullptr);

    constexpr bool use_logit_softcap = false;
    ggml_cuda_flash_attn_ext_vec_gqa_case_impl<D, GQA_PACK, type_K, type_V, use_logit_softcap>(ctx, dst);
}

#define DECL_FATTN_VEC_GQA_CASE(D, GQA_PACK, type_K, type_V)                        \
    template void ggml_cuda_flash_attn_ext_vec_gqa_case                             \
    <D, GQA_PACK, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_GQA_CASES(D, type_K, type_V)     \
    extern DECL_FATTN_VEC_GQA_CASE(D, 2, type_K, type_V);      \
    extern DECL_FATTN_VEC_GQA_CASE(D, 6, type_K, type_V);      \

EXTERN_DECL_FATTN_VEC_GQA_CASES(256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
