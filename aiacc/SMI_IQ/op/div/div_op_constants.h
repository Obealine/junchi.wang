#ifndef DIV_OP_CONSTANTS_H
#define DIV_OP_CONSTANTS_H

/**
 * @file div_op_constants.h
 * @brief Div 算子常量：RtoVR 端口 + Newton 迭代 + RTL 写回时序
 */

#include <cstdint>

namespace aiacc {
namespace op {
namespace div {

constexpr uint8_t VLD_BIT = 0x80;

constexpr uint8_t RTOVR_IN_DR_FPMUL_D   = 3;
constexpr uint8_t RTOVR_IN_DR_FPADD_D   = 4;
constexpr uint8_t RTOVR_IN_UR_RDATA_0   = 30;
constexpr uint8_t RTOVR_IN_UR_RDATA_1   = 31;
constexpr uint8_t RTOVR_IN_UR_RDATA_2   = 32;
constexpr uint8_t RTOVR_IN_UR_RDATA_3   = 33;

constexpr uint8_t RTOVR_OUT_DVR_FPMUL_S0   = 3;
constexpr uint8_t RTOVR_OUT_DVR_FPMUL_S1   = 4;
constexpr uint8_t RTOVR_OUT_DVR_FPADD_S0   = 5;
constexpr uint8_t RTOVR_OUT_DVR_FPADD_S1   = 6;
constexpr uint8_t RTOVR_OUT_UVR_WDATA_0    = 35;

constexpr uint32_t FPMUL_LATENCY  = 4;
/// RTL：FPMUL issue 后第 3 个 emit 拍 dr_fpmul_d 有效（v_pipe[3]）。
constexpr uint32_t FPMUL_DR_VALID_EMITS_AFTER_ISSUE = 3u;
constexpr uint32_t FPADD_LATENCY  = 4;
constexpr uint32_t BATCH          = 1;

/// 含 B+10 写回 FPMUL；步内 rv/ur 对齐 CPI。
constexpr uint32_t CYCLES_PER_ITERATION = 19;
constexpr uint32_t RTOVR_OUTPUT_REG_CYCLES = 1u;
constexpr uint32_t UVR_WDATA_SETTLE_AFTER_RTOVR = 1u;
constexpr uint32_t WB_UR_CYCLES_AFTER_RTOVR =
    RTOVR_OUTPUT_REG_CYCLES + UVR_WDATA_SETTLE_AFTER_RTOVR;

constexpr uint32_t CYCLES_FINAL_MUL = 19;
constexpr uint32_t CYCLES_FINAL_TAIL_EXTRA = 0;
constexpr uint32_t DEFAULT_NEWTON_ITERS = 3;

constexpr uint32_t RTL_IQ_DEPTH_LIMIT = 32;

inline uint64_t div_est_ur_pair_iq_inst_slots(uint32_t total_steps, uint32_t newton_iters) {
    if (newton_iters == 0u)
        newton_iters = DEFAULT_NEWTON_ITERS;
    if (total_steps <= 1u)
        return 6ull;
    const uint64_t events = static_cast<uint64_t>(total_steps) *
                            (static_cast<uint64_t>(newton_iters) + 1u);
    return events * 9ull;
}

inline uint64_t div_est_ur_pair_linear_inst_slots(uint32_t total_steps, uint32_t newton_iters) {
    return div_est_ur_pair_iq_inst_slots(total_steps, newton_iters);
}

inline uint32_t clamp_div_newton_iters_for_rtl_iq(uint32_t total_steps, uint32_t requested_iters) {
    uint32_t i = requested_iters;
    if (i == 0)
        i = DEFAULT_NEWTON_ITERS;
    if (i > 4)
        i = 4;
    for (; i >= 1u; --i) {
        const uint64_t slots = static_cast<uint64_t>(total_steps) * (static_cast<uint64_t>(i) + 1u);
        if (slots * 6ull > static_cast<uint64_t>(RTL_IQ_DEPTH_LIMIT))
            continue;
        if (div_est_ur_pair_linear_inst_slots(total_steps, i) >
            static_cast<uint64_t>(RTL_IQ_DEPTH_LIMIT))
            continue;
        return i;
    }
    return 1u;
}

inline uint32_t compute_total_cycles(uint32_t iters) {
    if (iters == 0) iters = DEFAULT_NEWTON_ITERS;
    return iters * CYCLES_PER_ITERATION + CYCLES_FINAL_MUL + CYCLES_FINAL_TAIL_EXTRA;
}

} // namespace div
} // namespace op
} // namespace aiacc

#endif // DIV_OP_CONSTANTS_H
