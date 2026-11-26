#ifndef DIV_OP_H
#define DIV_OP_H

/**
 * @file div_op.h
 * @brief Div 算子：out = input / other (FP32, Newton-Raphson)
 *
 * ============================================================================
 * 数学公式
 * ============================================================================
 *
 * Newton-Raphson 求 1/D:
 *   x_{n+1} = x_n * (2 - D * x_n)
 * 最终:
 *   result = N * x_final
 *
 * 初值 x_0 由 magic number 给出：
 *   i = 0x7EF311C7 - *(uint32_t*)&D
 *
 * ============================================================================
 * 流水线设计
 * ============================================================================
 *
 * 每次 Newton 迭代 = 16 cycles:
 *   2 次 FPMUL (D*x_n, x_n*(2-D*x_n))
 *   1 次 FPADD_SUB (2 - D*x_n)
 *   UR 读写 + RtoVR 路由
 *
 * 最终乘法 = 7 cycles:
 *   1 次 FPMUL (N * x_final)
 *
 * 硬件参数：
 *   - FPMUL 流水线级数: 4
 *   - FPADD 流水线级数: 4
 *   - 牛顿迭代次数: 可配置 (默认 3)
 *
 * ============================================================================
 * UR Bank 分配
 * ============================================================================
 *
 *   port 0: D (denominator) / N (numerator, final step)
 *   port 1: x_n (当前估计值)
 *   port 2: const 2.0
 *   write port 0: x_{n+1} / result
 *
 * ============================================================================
 */

#include <cstdint>
#include <vector>
#include <string>
#include <random>

#include "../../hal/aiacc_smi_iq.h"
#include "../../iq/modules/ur_iq.h"
#include "../../iq/modules/fpmul_iq.h"
#include "../../iq/modules/fpadd_iq.h"
#include "../common/op_instructions_base.h"
#include "../common/sched_entry.h"
#include "../common/op_base.h"
#include "../common/op_registry.h"
#include "../../app/app_json_parser.h"
#include "../../app/fp16_utils.h"

namespace aiacc {
namespace op {

// ==========================================
// Div 算子参数
// ==========================================

struct DivOpParams {
    // 张量维度
    uint32_t para_n{1};
    uint32_t para_c{1};
    uint32_t para_h{1};
    uint32_t para_w{4};

    // 数据类型: 0=FP32, 1=FP16
    uint8_t para_dtype{0};

    // 牛顿迭代次数（默认 3 次）
    uint32_t para_iters{3};

    // ===== UR Bank 配置 =====
    // D (denominator) — UR read port 0
    uint8_t para_D_bank{0};
    uint32_t para_D_addr{0};

    // N (numerator) — UR read port 0 (final step only)
    uint8_t para_N_bank{1};
    uint32_t para_N_addr{0};

    // x0 (initial guess) — UR read port 1 (first iter)
    uint8_t para_x0_bank{2};
    uint32_t para_x0_addr{0};

    // const 2.0 — UR read port 2
    uint8_t para_const2_bank{3};
    uint32_t para_const2_addr{0};

    // output（与 tvgen default 及 compile 缺省一致：bank4）
    uint8_t para_output_bank{4};
    uint32_t para_output_addr{0};

    // total elements needed for dynamic pointer offsetting
    uint32_t para_total_elements{0};

    // 辅助函数
    uint32_t get_elems_per_word() const {
        return (para_dtype == 0) ? 4 : 8;  // FP32: 4, FP16: 8
    }
};

// ==========================================
// Div 算子微指令序列 (schedule-based)
//
//   每个 IQ 维护一个 SchedEntry 列表 (abs_cycle, code)。
//   get_iq_files() 中将 abs_cycle 差值转换为 DLY 字段。
// ==========================================

struct DivOpInstructions : public OpInstructionsBase {
    // ==========================================
    // 各 IQ 的调度条目
    // ==========================================
    std::vector<SchedEntry> sched_ur_r0;     // UR read port 0 (D / N)
    std::vector<SchedEntry> sched_ur_r1;     // UR read port 1 (D)
    std::vector<SchedEntry> sched_ur_r2;     // UR read port 2 (x_n / x_final)
    std::vector<SchedEntry> sched_ur_r3;     // UR read port 3 (const 2.0)
    std::vector<SchedEntry> sched_ur_w0;     // UR write port 0 (x_{n+1} / result)
    std::vector<SchedEntry> sched_fpmul;     // FPMUL
    std::vector<SchedEntry> sched_fpadd;     // FPADD(SUB)
    std::vector<SchedEntry> sched_rtovr_3;   // RtoVR port 3  (FPMUL S0)
    std::vector<SchedEntry> sched_rtovr_4;   // RtoVR port 4  (FPMUL S1)
    std::vector<SchedEntry> sched_rtovr_5;   // RtoVR port 5  (FPADD S0)
    std::vector<SchedEntry> sched_rtovr_6;   // RtoVR port 6  (FPADD S1)
    std::vector<SchedEntry> sched_rtovr_35;  // RtoVR port 35 (UVR WDATA 0)

    // ==========================================
    // 计算单元 CODE 值
    // ==========================================
    uint8_t para_fpmul_code{0};  // FPMUL CODE (3 bits)
    uint8_t para_fpadd_code{0};  // FPADD CODE (5 bits, is_sub=1)

    // ==========================================
    // 统计信息
    // ==========================================
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_newton_iters{3};
    uint32_t para_total_cycles{0};
    uint32_t para_step_span{0};

    // 实现基类接口
    std::vector<IqInstructionFile> get_iq_files() const override;
    std::string get_operator_name() const override { return "Div"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
};

// ==========================================
// Div 算子编译器
// ==========================================

class DivOp : public OpBase {
public:
    DivOp() = default;
    ~DivOp() = default;

    // 实现 OpBase 接口
    std::string get_name() const override;
    OpInstructionsBase* compile(const app::AppOperatorRequest& req) override;
    bool calculate_golden(const app::AppOperatorRequest& req,
                         std::vector<uint8_t>& golden_bin) override;
    bool generate_data(const app::AppOperatorRequest& req,
                      std::vector<uint8_t>& input_bin,
                      std::vector<uint8_t>& other_bin,
                      std::vector<uint8_t>& alpha_bin) override;
    void fill_random_request(app::AppOperatorRequest& req,
                             std::mt19937& rng,
                             std::uniform_real_distribution<float>& data_dist,
                             const std::string& dtype_str) override;

    // 编译方法
    DivOpInstructions compile(DivOpParams& params);

private:
    // ==========================================
    // 计算单元指令生成 (delegates to div_op_compute_inst.cpp)
    // ==========================================
    void gen_fpmul_inst(const DivOpParams& params, iq::FpMulCru& cru);
    void gen_fpadd_sub_inst(const DivOpParams& params, iq::FpAddCru& cru);
};

// ==========================================
// 辅助函数
// ==========================================

void print_div_instructions(const DivOpInstructions& insts);

} // namespace op
} // namespace aiacc

#endif // DIV_OP_H
