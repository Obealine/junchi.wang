#include "div_op.h"
#include "div_op_constants.h"
#include "div_op_compute_inst.h"
#include "../common/ur_iq_serialize.h"
#include "../common/ur_iq_rtl_emit.h"
#include "../../app/fp16_utils.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <string>
#include <fstream>
#include <chrono>
#include "../../iq/iq_id_defs.h"

namespace aiacc {
namespace op {

using namespace div;

namespace {

// #region agent log
static void agent_debug_log(const char* location, const char* message, const char* hypothesis_id,
                            const char* run_id, const std::string& data_json) {
    std::ofstream f("/home/oliver/Oliver/.cursor/debug-5241f9.log", std::ios::app);
    if (!f) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    f << "{\"sessionId\":\"5241f9\",\"location\":\"" << location << "\",\"message\":\"" << message
      << "\",\"hypothesisId\":\"" << hypothesis_id << "\",\"runId\":\"" << run_id
      << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
}
// #endregion

static bool realign_div_wb_from_fpmul(DivOpInstructions& result);

int32_t read_i32_param(const app::AppOperatorRequest& req, const char* key, int32_t default_val) {
    auto it = req.para_parameters.find(key);
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty())
        return default_val;
    return it->second.para_data_i32[0];
}

uint32_t read_u32_addr_param(const app::AppOperatorRequest& req, const char* key, uint32_t default_val) {
    int32_t v = read_i32_param(req, key, static_cast<int32_t>(default_val));
    if (v < 0)
        return default_val;
    return static_cast<uint32_t>(v);
}

uint8_t read_bank_param(const app::AppOperatorRequest& req, const char* key, uint8_t default_bank) {
    int32_t v = read_i32_param(req, key, static_cast<int32_t>(default_bank));
    if (v < 0 || v > 7)
        return default_bank;
    return static_cast<uint8_t>(v);
}

} // namespace

static inline uint32_t ur_read_code(uint8_t bank, uint32_t addr) {
    return 1u | ((uint32_t)(bank & 0x7) << 1) | ((addr & 0xFFu) << 4);
}

static inline uint32_t ur_write_code(uint8_t bank, uint32_t addr, uint16_t wbe = 0xFFFF) {
    return 1u | ((uint32_t)(bank & 0x7) << 1) | ((addr & 0xFFu) << 4) | ((uint32_t)(wbe & 0xFFFF) << 12);
}

static inline uint32_t rtovr_code(uint8_t source_id) {
    return 1u | ((uint32_t)(source_id & 0x3F) << 1);
}

std::string DivOp::get_name() const { return "Div"; }

OpInstructionsBase* DivOp::compile(const app::AppOperatorRequest& req) {
    DivOpParams params;
    params.para_n = req.para_dimensions.para_n;
    params.para_c = req.para_dimensions.para_c;
    params.para_h = req.para_dimensions.para_h;
    params.para_w = req.para_dimensions.para_w;
    if (params.para_n == 0 || params.para_c == 0 ||
        params.para_h == 0 || params.para_w == 0)
        throw std::runtime_error("Invalid dimensions");

    if (!req.para_inputs.empty()) {
        const auto& first = req.para_inputs.begin()->second;
        params.para_dtype = (first.para_dtype == app::AppDataType::FLOAT32) ? 0 : 1;
    } else {
        params.para_dtype = 0;
    }

    params.para_total_elements = params.para_n * params.para_c * params.para_h * params.para_w;

    // 默认与 tvgen_core default 布局一致：bank0 N，bank1 D，bank2 x0，bank3 const2，bank4 输出。
    params.para_N_bank = 0;
    params.para_N_addr = 0;
    params.para_D_bank = 1;
    params.para_D_addr = 0;
    params.para_x0_bank = 2;
    params.para_x0_addr = 0;
    params.para_const2_bank = 3;
    params.para_const2_addr = 0;
    params.para_output_bank = 4;
    params.para_output_addr = 0;

    // JSON / 命令行 int 覆盖（与 fill_random_request 写入的 N_bank、input_addr 等）可选改写 UR 端口映射。
    params.para_N_bank = read_bank_param(req, "N_bank", params.para_N_bank);
    params.para_D_bank = read_bank_param(req, "D_bank", params.para_D_bank);
    params.para_x0_bank = read_bank_param(req, "x0_bank", params.para_x0_bank);
    params.para_const2_bank = read_bank_param(req, "const2_bank", params.para_const2_bank);
    params.para_output_bank = read_bank_param(req, "output_bank", params.para_output_bank);
    params.para_N_addr = read_u32_addr_param(req, "N_addr", params.para_N_addr);
    params.para_D_addr = read_u32_addr_param(req, "D_addr", params.para_D_addr);
    params.para_x0_addr = read_u32_addr_param(req, "x0_addr", params.para_x0_addr);
    params.para_const2_addr = read_u32_addr_param(req, "const2_addr", params.para_const2_addr);
    params.para_output_addr = read_u32_addr_param(req, "output_addr", params.para_output_addr);

    auto iters_it = req.para_parameters.find("iters");
    if (iters_it != req.para_parameters.end())
        if (!iters_it->second.para_data_i32.empty())
            params.para_iters = static_cast<uint32_t>(iters_it->second.para_data_i32[0]);
    auto iters_it2 = req.para_parameters.find("newton_iters");
    if (iters_it2 != req.para_parameters.end())
        if (!iters_it2->second.para_data_i32.empty())
            params.para_iters = static_cast<uint32_t>(iters_it2->second.para_data_i32[0]);

    DivOpInstructions* insts = new DivOpInstructions();
    *insts = compile(params);
    return insts;
}

DivOpInstructions DivOp::compile(DivOpParams& params) {
    if (params.para_n == 0 || params.para_c == 0 ||
        params.para_h == 0 || params.para_w == 0)
        throw std::runtime_error("Invalid dimensions");

    const uint32_t total_elems = params.para_n * params.para_c *
                                  params.para_h * params.para_w;
    const uint32_t elems_per_word = params.get_elems_per_word();
    const uint32_t total_steps = (total_elems + elems_per_word - 1) / elems_per_word;

    uint32_t try_iters = (params.para_iters == 0) ? DEFAULT_NEWTON_ITERS : params.para_iters;
    if (try_iters > 4)
        try_iters = 4;
    try_iters = div::clamp_div_newton_iters_for_rtl_iq(total_steps, try_iters);

    for (;;) {
        DivOpInstructions result;
        result.para_total_steps = total_steps;
        result.para_total_elements = total_elems;
        result.para_newton_iters = try_iters;

        {
            iq::FpMulCru fpmul_cru;
            gen_fpmul_inst(params, fpmul_cru);
            result.para_fpmul_code = iq::FpMulIq::pack_cru_fpmul(fpmul_cru);
        }
        {
            iq::FpAddCru fpadd_cru;
            gen_fpadd_sub_inst(params, fpadd_cru);
            result.para_fpadd_code = iq::FpAddIq::pack_cru_fpadd(fpadd_cru);
        }
        constexpr uint32_t CPI = CYCLES_PER_ITERATION;
        constexpr uint32_t CFM = CYCLES_FINAL_MUL;

        const uint32_t CPI_eff = CPI;
        const uint32_t CFM_eff = CFM + CYCLES_FINAL_TAIL_EXTRA;

        for (uint32_t step = 0; step < total_steps; ++step) {
            const uint32_t step_base = step * (try_iters * CPI_eff + CFM_eff);

            for (uint32_t it = 0; it < try_iters; ++it) {
                const uint32_t B = step_base + it * CPI_eff;
                const bool is_first = (it == 0);
                uint8_t xb;
                uint32_t xa;
                if (is_first) {
                    xb = params.para_x0_bank;
                    xa = params.para_x0_addr + step;
                } else {
                    xb = params.para_output_bank;
                    xa = params.para_output_addr + step;
                }

                uint8_t ob = params.para_output_bank;
                uint32_t oa = params.para_output_addr + step;

                result.sched_ur_r1.push_back({B + 0, ur_read_code(params.para_D_bank, params.para_D_addr + step)});
                result.sched_ur_r2.push_back({B + 0, ur_read_code(xb, xa)});

                result.sched_rtovr_3.push_back({B + 1, rtovr_code(RTOVR_IN_UR_RDATA_1)});
                result.sched_rtovr_4.push_back({B + 1, rtovr_code(RTOVR_IN_UR_RDATA_2)});

                result.sched_fpmul.push_back({B + 2, result.para_fpmul_code});
                result.sched_ur_r3.push_back({B + 4, ur_read_code(params.para_const2_bank, params.para_const2_addr)});

                result.sched_rtovr_5.push_back({B + 5, rtovr_code(RTOVR_IN_UR_RDATA_3)});
                result.sched_rtovr_6.push_back({B + 5, rtovr_code(RTOVR_IN_DR_FPMUL_D)});

                result.sched_fpadd.push_back({B + 6, result.para_fpadd_code});
                result.sched_ur_r2.push_back({B + 8, ur_read_code(xb, xa)});

                result.sched_rtovr_3.push_back({B + 9, rtovr_code(RTOVR_IN_UR_RDATA_2)});
                result.sched_rtovr_4.push_back({B + 9, rtovr_code(RTOVR_IN_DR_FPADD_D)});

                result.sched_fpmul.push_back({B + 10, result.para_fpmul_code});
                const uint32_t rv_wb = B + 10 + FPMUL_LATENCY;
                const uint32_t uw_wb = rv_wb + div::WB_UR_CYCLES_AFTER_RTOVR;
                result.sched_rtovr_35.push_back({rv_wb, rtovr_code(RTOVR_IN_DR_FPMUL_D)});
                result.sched_ur_w0.push_back({uw_wb, ur_write_code(ob, oa)});
            }

            const uint32_t F = step_base + try_iters * CPI_eff;
            uint8_t xf_bank = params.para_output_bank;
            uint32_t xf_addr = params.para_output_addr + step;

            result.sched_ur_r0.push_back({F + 0, ur_read_code(params.para_N_bank, params.para_N_addr + step)});
            result.sched_ur_r2.push_back({F + 0, ur_read_code(xf_bank, xf_addr)});
            result.sched_rtovr_3.push_back({F + 1, rtovr_code(RTOVR_IN_UR_RDATA_0)});
            result.sched_rtovr_4.push_back({F + 1, rtovr_code(RTOVR_IN_UR_RDATA_2)});
            result.sched_fpmul.push_back({F + 10, result.para_fpmul_code});
            const uint32_t rvf_wb = F + 10 + FPMUL_LATENCY;
            const uint32_t uwf_wb = rvf_wb + div::WB_UR_CYCLES_AFTER_RTOVR;
            result.sched_rtovr_35.push_back({rvf_wb, rtovr_code(RTOVR_IN_DR_FPMUL_D)});
            result.sched_ur_w0.push_back(
                {uwf_wb, ur_write_code(params.para_output_bank, params.para_output_addr + step)});
        }

        result.para_step_span = try_iters * CPI_eff + CFM_eff;
        result.para_total_cycles = total_steps * result.para_step_span;

        constexpr bool kRealignDivWbFromFpmul = true;
        if (kRealignDivWbFromFpmul && !realign_div_wb_from_fpmul(result)) {
            throw std::runtime_error("Div: writeback wall exceeds step boundary; increase CYCLES_PER_ITERATION");
        }

        try {
            (void)result.get_iq_files();
            params.para_iters = try_iters;
            return result;
        } catch (const std::runtime_error& e) {
            const std::string msg(e.what());
            if (msg.find("IQ depth") == std::string::npos && msg.find("depth=") == std::string::npos)
                throw;
            if (try_iters <= 1u) {
                throw std::runtime_error(
                    "Div: 展开后仍超过 Jianyuan 默认 IQ 深度(32)。请减小张量尺寸或降低 newton_iters。详情: " + msg);
            }
            --try_iters;
        }
    }
}

void DivOp::gen_fpmul_inst(const DivOpParams& params, iq::FpMulCru& cru) {
    div_gen_fpmul_inst(params, cru);
}
void DivOp::gen_fpadd_sub_inst(const DivOpParams& params, iq::FpAddCru& cru) {
    div_gen_fpadd_sub_inst(params, cru);
}

namespace {

constexpr uint8_t kMaxDlyField = 7;
// 与 Jianyuan `smi_iq` 默认 INSTQ_DEPTH=32、TB `inst_mem_*[0:31]` 一致（勿与未改 RTL 的仓库假设冲突）。
constexpr uint32_t kRtlIqDepth = div::RTL_IQ_DEPTH_LIMIT;
/// ur-w0 同址双写须 EMBED（RTL waddr=micro 槽）；步长取自 sched 实际 abs 间隔。
// RTL ur-w0：IIC 用 operate_micro_addr 作 waddr；同 bank/addr 重复写须 EMBED+inc_embed 递增 micro 槽。
constexpr bool kEnableUrWbEmbedRtlAddr = false;

struct IQHeader41 {
    uint8_t vld{1};
    uint8_t jump{0};
    uint8_t embed{0};
    uint8_t embed_end{0};
    uint16_t loops{0};
    uint8_t cond{0};
    uint8_t dly{0};
    int8_t inc{0};
    uint8_t inc_embed{0};
};

static uint64_t serialize_header_41(const IQHeader41& h) {
    uint64_t bits = 0;
    bits |= (uint64_t)(h.inc_embed & 0x3F) << 0;
    bits |= (uint64_t)(h.inc & 0xF) << 6;
    bits |= (uint64_t)(h.dly & 0x7) << 10;
    bits |= (uint64_t)(h.cond & 0x7F) << 13;
    bits |= (uint64_t)(h.loops & 0x3FF) << 20;
    bits |= (uint64_t)(h.embed_end & 0x3F) << 30;
    bits |= (uint64_t)(h.embed & 0x7) << 36;
    bits |= (uint64_t)(h.jump & 0x1) << 39;
    bits |= (uint64_t)(h.vld & 0x1) << 40;
    return bits;
}

static std::vector<uint8_t> pack_instruction(const IQHeader41& h, uint32_t code, uint32_t code_bits) {
    uint64_t header = serialize_header_41(h);
    uint32_t total_bits = 41 + code_bits;
    uint32_t num_bytes = (total_bits + 7) / 8;

    if (total_bits <= 64) {
        uint64_t code_mask = (code_bits < 64) ? ((1ULL << code_bits) - 1) : ~0ULL;
        uint64_t total = (header << code_bits) | ((uint64_t)code & code_mask);
        std::vector<uint8_t> bytes;
        for (int i = (int)num_bytes - 1; i >= 0; --i)
            bytes.push_back(static_cast<uint8_t>((total >> (i * 8)) & 0xFF));
        return bytes;
    } else {
        unsigned __int128 tv = (static_cast<unsigned __int128>(header) << code_bits) |
                               static_cast<unsigned __int128>(code & ((1u << code_bits) - 1u));
        std::vector<uint8_t> bytes;
        for (int i = (int)num_bytes - 1; i >= 0; --i)
            bytes.push_back(static_cast<uint8_t>((tv >> (i * 8)) & 0xFF));
        return bytes;
    }
}

static std::vector<uint8_t> pack_instruction_legacy(
    uint8_t vld, uint8_t jump, uint8_t dly, uint32_t code, uint32_t code_bits) {
    IQHeader41 h{};
    h.vld = vld;
    h.jump = jump;
    h.dly = dly;
    return pack_instruction(h, code, code_bits);
}

static std::vector<uint8_t> sched_to_iq_bytes(const std::vector<SchedEntry>& sched,
                                               uint32_t code_bits) {
    std::vector<uint8_t> data;
    int32_t prev_cycle = -1;

    for (const auto& entry : sched) {
        int32_t target_dly = ur_iq_serialize::iq_target_dly_from_abs(
            prev_cycle, static_cast<int32_t>(entry.abs_cycle));
        if (target_dly < 0)
            target_dly = 0;

        while (target_dly > (int32_t)kMaxDlyField) {
            auto nop = pack_instruction_legacy(1, 0, kMaxDlyField, 0, code_bits);
            data.insert(data.end(), nop.begin(), nop.end());
            target_dly -= 8;
        }

        auto inst = pack_instruction_legacy(1, 0, (uint8_t)target_dly, entry.code, code_bits);
        data.insert(data.end(), inst.begin(), inst.end());
        prev_cycle = static_cast<int32_t>(entry.abs_cycle);
    }

    auto end_inst = pack_instruction_legacy(1, 1, 0, 0, code_bits);
    data.insert(data.end(), end_inst.begin(), end_inst.end());
    return data;
}

/// RTL 多 IQ 对齐：abs_cycle 为芯片全局墙钟，发射拍 = abs+1（避免各 IQ 局部 prev 错位）。
static std::vector<uint8_t> sched_to_iq_bytes_rtl_global_wall(const std::vector<SchedEntry>& sched,
                                                               uint32_t code_bits) {
    std::vector<uint8_t> data;
    int32_t prev_global_wall = 0;

    for (const auto& entry : sched) {
        const int32_t target_wall = static_cast<int32_t>(entry.abs_cycle) + 1;
        int32_t target_dly = target_wall - prev_global_wall - 1;
        if (target_dly < 0) {
            target_dly = 0;
        }

        while (target_dly > (int32_t)kMaxDlyField) {
            auto nop = pack_instruction_legacy(1, 0, kMaxDlyField, 0, code_bits);
            data.insert(data.end(), nop.begin(), nop.end());
            target_dly -= 8;
            prev_global_wall += 8;
        }

        auto inst = pack_instruction_legacy(1, 0, (uint8_t)target_dly, entry.code, code_bits);
        data.insert(data.end(), inst.begin(), inst.end());
        prev_global_wall = target_wall;
    }

    auto end_inst = pack_instruction_legacy(1, 1, 0, 0, code_bits);
    data.insert(data.end(), end_inst.begin(), end_inst.end());
    return data;
}

static uint32_t sum_issue_cycles_packed(const std::vector<uint8_t>& data,
                                        uint32_t width_bits,
                                        uint32_t code_bits);

/// EMBED 体从 abs=0 打包后，在 IQ 首端补齐 base_abs 墙钟气泡，与 fpmul 等同起点的 global_wall 对齐。
static void prepend_rtl_iq_global_wall_offset(std::vector<uint8_t>& data,
                                            uint32_t offset_cycles,
                                            uint32_t code_bits) {
    if (offset_cycles == 0u || data.empty()) {
        return;
    }
    const uint32_t inst_bytes = (41u + code_bits + 7u) / 8u;
    if (inst_bytes == 0u) {
        return;
    }
    std::vector<uint8_t> prefix;
    uint32_t remain = offset_cycles;
    while (remain > 0u) {
        const uint32_t dly = std::min<uint32_t>(remain > 0u ? remain - 1u : 0u, kMaxDlyField);
        auto nop = pack_instruction_legacy(1, 0, static_cast<uint8_t>(dly), 0, code_bits);
        if (nop.size() != inst_bytes) {
            break;
        }
        prefix.insert(prefix.end(), nop.begin(), nop.end());
        remain -= dly + 1u;
    }
    if (!prefix.empty()) {
        data.insert(data.begin(), prefix.begin(), prefix.end());
    }
}

/// rtovr-35 与 ur-w0 成对：ur 比 rv 晚 WB_UR_CYCLES_AFTER_RTOVR（含锁存后 uvr 稳定）。
static void validate_paired_rtovr35_ur_w(const std::vector<SchedEntry>& rv,
                                        const std::vector<SchedEntry>& uw) {
    if (rv.size() != uw.size()) {
        throw std::runtime_error("Div: sched_rtovr_35 and sched_ur_w0 size mismatch");
    }
    for (size_t i = 0; i < rv.size(); ++i) {
        if (uw[i].abs_cycle != rv[i].abs_cycle + div::WB_UR_CYCLES_AFTER_RTOVR) {
            throw std::runtime_error(
                "Div: ur-w0 must be " + std::to_string(div::WB_UR_CYCLES_AFTER_RTOVR) +
                " cycles after rtovr-35 at index " + std::to_string(i));
        }
    }
    for (size_t i = 1; i < rv.size(); ++i) {
        const int32_t dr =
            static_cast<int32_t>(rv[i].abs_cycle) - static_cast<int32_t>(rv[i - 1].abs_cycle);
        const int32_t du =
            static_cast<int32_t>(uw[i].abs_cycle) - static_cast<int32_t>(uw[i - 1].abs_cycle);
        if (dr != du) {
            throw std::runtime_error(
                "Div: paired RtoVR-35 / ur-w0 abs_cycle delta mismatch at index " +
                std::to_string(i));
        }
    }
}

static void append_rtovr7_long_delay(std::vector<uint8_t>& data, uint32_t& total_dly) {
    auto zero_code = ur_iq_serialize::serialize_code(0u, 7);
    ur_iq_serialize::emit_dummy_for_long_delay(total_dly, [&]() {
        auto dummy = ur_iq_serialize::make_instruction(1, 0, 0, 0, 0, kMaxDlyField, 0, 0, 0,
                                                      zero_code, 7);
        data.insert(data.end(), dummy.begin(), dummy.end());
    });
}

static void append_ur_w28_long_delay(std::vector<uint8_t>& data, uint32_t& total_dly) {
    ur_iq_serialize::emit_dummy_for_long_delay(total_dly, [&]() {
        auto dummy = ur_iq_serialize::serialize_ur_wdata_inst(1, 0, 0, 0, 0, kMaxDlyField, 0, 0, 0, 0);
        data.insert(data.end(), dummy.begin(), dummy.end());
    });
}

/// compile 调度是否满足写回 EMBED 步长（内层 rv@B+14、ur@B+17，final rv@F+14）。
static bool validate_sched_for_mul_style_wb(const std::vector<SchedEntry>& rv,
                                            const std::vector<SchedEntry>& uw,
                                            uint32_t newton_iters,
                                            uint32_t total_steps,
                                            uint32_t step_span) {
    if (rv.size() != uw.size() || newton_iters == 0u || step_span == 0u)
        return false;
    const uint32_t per_step = newton_iters + 1u;
    if (rv.size() != total_steps * per_step)
        return false;

    const uint32_t cpi = div::CYCLES_PER_ITERATION;
    for (uint32_t step = 0; step < total_steps; ++step) {
        const uint32_t step_base = step * step_span;
        for (uint32_t it = 0; it < newton_iters; ++it) {
            const uint32_t idx = step * per_step + it;
            const uint32_t B = step_base + it * cpi;
            const uint32_t expect_rv = B + 10u + div::FPMUL_DR_VALID_EMITS_AFTER_ISSUE;
            if (rv[idx].abs_cycle != expect_rv ||
                uw[idx].abs_cycle != expect_rv + div::WB_UR_CYCLES_AFTER_RTOVR)
                return false;
        }
        const uint32_t idx_f = step * per_step + newton_iters;
        const uint32_t F = step_base + newton_iters * cpi;
        const uint32_t expect_rvf = F + 10u + div::FPMUL_DR_VALID_EMITS_AFTER_ISSUE;
        if (rv[idx_f].abs_cycle != expect_rvf ||
            uw[idx_f].abs_cycle != expect_rvf + div::WB_UR_CYCLES_AFTER_RTOVR)
            return false;
    }
    return true;
}

static std::vector<uint8_t> sched_to_iq_bytes_no_end(const std::vector<SchedEntry>& sched,
                                                      uint32_t code_bits) {
    std::vector<uint8_t> data;
    int32_t prev_cycle = -1;

    for (const auto& entry : sched) {
        int32_t target_dly = ur_iq_serialize::iq_target_dly_from_abs(
            prev_cycle, static_cast<int32_t>(entry.abs_cycle));
        if (target_dly < 0)
            target_dly = 0;

        while (target_dly > (int32_t)kMaxDlyField) {
            auto nop = pack_instruction_legacy(1, 0, kMaxDlyField, 0, code_bits);
            data.insert(data.end(), nop.begin(), nop.end());
            target_dly -= 8;
        }

        auto inst = pack_instruction_legacy(1, 0, (uint8_t)target_dly, entry.code, code_bits);
        data.insert(data.end(), inst.begin(), inst.end());
        prev_cycle = static_cast<int32_t>(entry.abs_cycle);
    }
    return data;
}

static bool build_step_inc_embed_flags(
    const std::vector<SchedEntry>& sched, uint32_t body_len, std::vector<bool>& flags) {
    flags.assign(body_len, false);
    if (body_len == 0 || sched.size() < body_len * 2) return true;
    const uint32_t steps = static_cast<uint32_t>(sched.size() / body_len);
    for (uint32_t i = 0; i < body_len; ++i) {
        const uint32_t base = sched[i].code;
        bool is_plus_addr = true;
        bool is_const = true;
        for (uint32_t s = 1; s < steps; ++s) {
            const uint32_t code = sched[s * body_len + i].code;
            if (code != (base + (s << 4))) is_plus_addr = false;
            if (code != base) is_const = false;
        }
        if (!is_plus_addr && !is_const) return false;
        flags[i] = is_plus_addr;
    }
    return true;
}

/// 与 `sched_to_iq_bytes_embed_loop` 前置条件一致；用于在打包前判断是否走 EMBED（避免 ur-w/rtovr-35 一路嵌入一路线性）。
static bool sched_embed_loop_eligible(const std::vector<SchedEntry>& sched,
                                      uint32_t total_steps,
                                      uint32_t step_span) {
    if (sched.empty() || total_steps <= 1 || step_span == 0)
        return false;
    if (sched.size() % total_steps != 0)
        return false;
    const uint32_t body_len = static_cast<uint32_t>(sched.size() / total_steps);
    std::vector<bool> inc_embed_flags;
    if (body_len == 0 || !build_step_inc_embed_flags(sched, body_len, inc_embed_flags))
        return false;
    for (uint32_t step = 1; step < total_steps; ++step) {
        for (uint32_t i = 0; i < body_len; ++i) {
            const auto& base = sched[i];
            const auto& cur = sched[step * body_len + i];
            if (cur.abs_cycle != base.abs_cycle + step * step_span)
                return false;
        }
    }
    return true;
}

static std::vector<uint8_t> sched_to_iq_bytes_embed_loop(
    const std::vector<SchedEntry>& sched,
    uint32_t code_bits,
    uint32_t total_steps,
    uint32_t step_span,
    bool enable_inc,
    bool rtl_single_body_ur_breaker = false
) {
    if (sched.empty() || total_steps <= 1 || step_span == 0) {
        return sched_to_iq_bytes(sched, code_bits);
    }
    if (sched.size() % total_steps != 0) {
        return sched_to_iq_bytes(sched, code_bits);
    }

    const uint32_t body_len = static_cast<uint32_t>(sched.size() / total_steps);
    std::vector<bool> inc_embed_flags;
    if (body_len == 0 || !build_step_inc_embed_flags(sched, body_len, inc_embed_flags)) {
        return sched_to_iq_bytes(sched, code_bits);
    }

    for (uint32_t step = 1; step < total_steps; ++step) {
        for (uint32_t i = 0; i < body_len; ++i) {
            const auto& base = sched[i];
            const auto& cur = sched[step * body_len + i];
            if (cur.abs_cycle != base.abs_cycle + step * step_span) {
                return sched_to_iq_bytes(sched, code_bits);
            }
        }
    }

    struct BodyInst { uint32_t dly; uint32_t code; bool apply_inc_embed; };
    std::vector<BodyInst> body;
    int32_t prev_cycle = -1;
    for (uint32_t i = 0; i < body_len; ++i) {
        const auto& entry = sched[i];
        int32_t target_dly = ur_iq_serialize::iq_target_dly_from_abs(
            prev_cycle, static_cast<int32_t>(entry.abs_cycle));
        if (target_dly < 0)
            target_dly = 0;
        while (target_dly > (int32_t)kMaxDlyField) {
            body.push_back({kMaxDlyField, 0, false});
            target_dly -= 8;
        }
        body.push_back({static_cast<uint32_t>(target_dly), entry.code, inc_embed_flags[i]});
        prev_cycle = static_cast<int32_t>(entry.abs_cycle);
    }

    // ur-w0 写指令在 EMBED 重复时 CODE 相同会触发 RTL inst_keep 多拍写；插入 we=0 的 breaker 打断。
    // SMC：body_len>1 才插 breaker；RTL 导出：body_len==1 同址双写也须 breaker（SMC 不读 *_rtl hex）。
    if (code_bits == 28u && !body.empty() && (body.back().code & 1u) &&
        (body_len > 1u || rtl_single_body_ur_breaker)) {
        body.push_back({0u, 2u, false});
    }

    // 在 EMBED 循环体尾部补齐 step 边界间隙，避免不同 step 被提前折叠执行。
    // 除 abs 对齐外，还须保证 body 内 DLY 链总墙钟 = step_span（body_len=1 时 last_abs 会少 1 拍 tail）。
    int32_t body_wall = 0;
    for (const auto& b : body) {
        body_wall += static_cast<int32_t>(b.dly) + 1;
    }
    const int32_t last_abs = static_cast<int32_t>(sched[body_len - 1].abs_cycle);
    int32_t tail_extra = static_cast<int32_t>(step_span) - last_abs - 1;
    const int32_t tail_from_wall = static_cast<int32_t>(step_span) - body_wall;
    if (tail_from_wall > tail_extra) {
        tail_extra = tail_from_wall;
    }
    while (tail_extra > 0) {
        const uint32_t pad_dly = static_cast<uint32_t>(std::min<int32_t>(kMaxDlyField, tail_extra - 1));
        body.push_back({pad_dly, 0, false});
        tail_extra -= static_cast<int32_t>(pad_dly) + 1;
    }

    std::vector<uint8_t> data;

    IQHeader41 ctrl{};
    ctrl.vld = 1;
    ctrl.embed = 1;
    ctrl.loops = static_cast<uint16_t>(total_steps - 1);
    ctrl.inc = enable_inc ? 1 : 0;
    auto ctrl_bytes = pack_instruction(ctrl, 0, code_bits);
    data.insert(data.end(), ctrl_bytes.begin(), ctrl_bytes.end());

    for (size_t i = 0; i < body.size(); ++i) {
        IQHeader41 h{};
        // DLY padding 须 vld=1 才能消耗气泡（与 ur-rdata EMBED / 线性 dummy 一致；SMC 在 vld=0 时不等待 DLY）。
        // code=0 时 we=0，ur-w0 不会误写 bank0。
        h.vld = 1u;
        h.dly = static_cast<uint8_t>(body[i].dly);
        h.embed_end = (i + 1 == body.size()) ? 0x01 : 0x00;
        h.inc_embed = (enable_inc && body[i].apply_inc_embed) ? 0x01 : 0x00;
        auto inst_bytes = pack_instruction(h, body[i].code, code_bits);
        data.insert(data.end(), inst_bytes.begin(), inst_bytes.end());
    }

    auto end_inst = pack_instruction_legacy(1, 1, 0, 0, code_bits);
    data.insert(data.end(), end_inst.begin(), end_inst.end());
    return data;
}

/// ur-w0 线性展开：每次 we=1 写后插入 code=2 breaker，避免 RTL smi_iq_dly_queen inst_keep。
static std::vector<SchedEntry> expand_ur_wb_sched_rtl_breakers(
    const std::vector<SchedEntry>& sched) {
    std::vector<SchedEntry> out;
    out.reserve(sched.size() * 2u);
    for (const auto& e : sched) {
        out.push_back(e);
        if (e.code & 1u) {
            out.push_back({e.abs_cycle + 1u, 2u});
        }
    }
    return out;
}

/// rtovr-35：每次 vld=1 路由后插入 code=0 breaker，打断 inst_keep 拖尾写 0。
static std::vector<SchedEntry> expand_rtovr_wb_sched_rtl_breakers(
    const std::vector<SchedEntry>& sched) {
    std::vector<SchedEntry> out;
    out.reserve(sched.size() * 2u);
    for (const auto& e : sched) {
        out.push_back(e);
        if (e.code & 1u) {
            out.push_back({e.abs_cycle + 1u, 0u});
        }
    }
    return out;
}

/// 按 fpmul@B+10 / F+10 的 RTL 墙钟重定位写回（含 step-EMBED）；abs≈墙钟（线性 DLY 域）。
static bool realign_div_wb_from_fpmul(DivOpInstructions& result) {
    if (result.sched_rtovr_35.size() != result.sched_ur_w0.size() ||
        result.sched_rtovr_35.empty()) {
        return true;
    }

    const uint32_t cpi = div::CYCLES_PER_ITERATION;
    const uint32_t cfm = div::CYCLES_FINAL_MUL + div::CYCLES_FINAL_TAIL_EXTRA;
    const uint32_t ni = result.para_newton_iters;
    const uint32_t per_step = ni + 1u;
    const uint32_t step_span = result.para_step_span;
    const uint32_t lat = div::FPMUL_LATENCY;
    // 含 rtovr-35 breaker 一拍：ur 须在 route 后 WB_UR_CYCLES（reg+settle），勿仅用 UVR_WDATA_SETTLE。
    const uint32_t ur_gap = div::WB_UR_CYCLES_AFTER_RTOVR;
    const uint32_t embed_steps = result.para_total_steps;

    for (size_t i = 0; i < result.sched_rtovr_35.size(); ++i) {
        const uint32_t step = static_cast<uint32_t>(i) / per_step;
        const uint32_t slot = static_cast<uint32_t>(i) % per_step;
        const uint32_t step_base = step * step_span;

        uint32_t fpmul_abs = 0;
        uint32_t cycle_hi = 0;
        if (slot < ni) {
            fpmul_abs = step_base + slot * cpi + 10u;
            cycle_hi = step_base + slot * cpi + cpi - 1u;
        } else {
            fpmul_abs = step_base + ni * cpi + 10u;
            cycle_hi = step_base + ni * cpi + cfm - 1u;
        }

        // fpmul/ur-w0/rtovr-35 均走 sched_to_iq_bytes_rtl_global_wall：emit = abs+1。
        const uint32_t issue_emit = fpmul_abs + 1u;
        (void)embed_steps;
        (void)step_span;

        // RTL：dr 在 issue_emit+FPMUL_DR_VALID（v_pipe[3]）；勿用 abs 域 LATENCY=4 直接加 emit。
        const uint32_t route_emit = issue_emit + div::FPMUL_DR_VALID_EMITS_AFTER_ISSUE;
        const uint32_t ur_emit = route_emit + ur_gap;
        const uint32_t abs_rv = route_emit - 1u;
        const uint32_t abs_ur = ur_emit - 1u;
        if (abs_rv > cycle_hi || abs_ur > cycle_hi) {
            return false;
        }

        result.sched_rtovr_35[i].abs_cycle = abs_rv;
        result.sched_ur_w0[i].abs_cycle = abs_ur;
    }
    return true;
}

/// RTL 线性 lower：sched.abs_cycle=墙钟；写后插 breaker 打断 inst_keep。
static std::vector<uint8_t> pack_div_wb_rtl_linear(const std::vector<SchedEntry>& sched,
                                                   uint32_t code_bits,
                                                   bool insert_ur_breakers,
                                                   bool insert_rtovr_breakers) {
    if (sched.empty()) {
        return {};
    }
    std::vector<SchedEntry> lowered = sched;
    if (insert_rtovr_breakers) {
        lowered = expand_rtovr_wb_sched_rtl_breakers(lowered);
    }
    if (insert_ur_breakers) {
        lowered = expand_ur_wb_sched_rtl_breakers(lowered);
    }
    return sched_to_iq_bytes_rtl_global_wall(lowered, code_bits);
}

/// 写回 EMBED 步长：相邻 embed 步首 body 槽的 abs 差（单元素 = 步内两次写回间隔）。
static uint32_t calc_div_wb_embed_span(const std::vector<SchedEntry>& sched,
                                     uint32_t newton_iters) {
    const uint32_t wb_steps = newton_iters + 1u;
    if (wb_steps <= 1u || sched.size() < wb_steps)
        return 0u;
    if (sched.size() % wb_steps != 0u)
        return 0u;
    const uint32_t body_len = static_cast<uint32_t>(sched.size() / wb_steps);
    if (body_len == 0u || sched.size() <= body_len)
        return 0u;
    const uint32_t span = sched[body_len].abs_cycle - sched[0].abs_cycle;
    return span > 0u ? span : 0u;
}

/// 写回 EMBED 前置：rv/ur 各 (newton_iters+1) 条，步长为 sched 实测间隔。
static bool sched_wb_embed_eligible(const std::vector<SchedEntry>& sched,
                                    uint32_t newton_iters) {
    const uint32_t wb_steps = newton_iters + 1u;
    const uint32_t wb_span = calc_div_wb_embed_span(sched, newton_iters);
    if (wb_span == 0u)
        return false;
    return sched_embed_loop_eligible(sched, wb_steps, wb_span);
}

/// ur-w0 同址写回：abs 归一化 EMBED（与 SMC 一致）；ur 比 rtovr 晚 UVR_WDATA_SETTLE 拍。
static std::vector<uint8_t> pack_div_wb_embed_rtl_global_wall(const std::vector<SchedEntry>& sched,
                                                              uint32_t code_bits,
                                                              uint32_t wb_steps,
                                                              uint32_t wb_span,
                                                              bool enable_inc,
                                                              bool rtl_single_body_ur_breaker) {
    if (sched.empty() || wb_steps <= 1u || wb_span == 0u ||
        sched.size() % wb_steps != 0u) {
        return {};
    }
    const uint32_t base_abs = sched[0].abs_cycle;
    std::vector<SchedEntry> norm;
    norm.reserve(sched.size());
    for (const auto& e : sched) {
        if (e.abs_cycle < base_abs) {
            return {};
        }
        norm.push_back({e.abs_cycle - base_abs, e.code});
    }
    auto data = sched_to_iq_bytes_embed_loop(
        norm, code_bits, wb_steps, wb_span, enable_inc, rtl_single_body_ur_breaker);
    if (!data.empty() && base_abs > 0u) {
        prepend_rtl_iq_global_wall_offset(data, base_abs, code_bits);
        // #region agent log
        agent_debug_log(
            "motion/div_op.cpp:pack_div_wb_embed_rtl_global_wall",
            "embed_rtl_wall_prepend",
            "G",
            "pre-fix",
            std::string("{\"base_abs\":") + std::to_string(base_abs) +
                ",\"code_bits\":" + std::to_string(code_bits) +
                ",\"cyc_after_prepend\":" +
                std::to_string(sum_issue_cycles_packed(data, 41u + code_bits, code_bits)) + "}");
        // #endregion
    }
    return data;
}

/// ur/rtovr 写回：EMBED 循环（步长 = sched 实测），与 fpmul/fpadd 同类压缩。
static std::vector<uint8_t> pack_div_wb_embed_loop(const std::vector<SchedEntry>& sched,
                                                   uint32_t code_bits,
                                                   uint32_t newton_iters,
                                                   bool enable_inc,
                                                   bool rtl_single_body_ur_breaker = false) {
    const uint32_t wb_steps = newton_iters + 1u;
    const uint32_t wb_span = calc_div_wb_embed_span(sched, newton_iters);
    if (wb_span == 0u || !sched_embed_loop_eligible(sched, wb_steps, wb_span)) {
        return sched_to_iq_bytes_rtl_global_wall(sched, code_bits);
    }
    auto rtl_embed = pack_div_wb_embed_rtl_global_wall(
        sched, code_bits, wb_steps, wb_span, enable_inc, rtl_single_body_ur_breaker);
    if (!rtl_embed.empty()) {
        return rtl_embed;
    }
    return sched_to_iq_bytes_embed_loop(
        sched, code_bits, wb_steps, wb_span, enable_inc, rtl_single_body_ur_breaker);
}

static std::vector<SchedEntry> normalize_sched_from_cycle(const std::vector<SchedEntry>& sched, uint32_t base_cycle) {
    std::vector<SchedEntry> normalized;
    normalized.reserve(sched.size());
    for (const auto& e : sched) {
        if (e.abs_cycle < base_cycle) {
            return {};
        }
        normalized.push_back({e.abs_cycle - base_cycle, e.code});
    }
    return normalized;
}

static std::vector<uint8_t> sched_to_iq_bytes_embed_loop_split_first(
    const std::vector<SchedEntry>& sched,
    uint32_t code_bits,
    uint32_t total_steps,
    uint32_t step_span,
    bool enable_inc = true
) {
    if (sched.empty() || total_steps <= 1 || step_span == 0) {
        return sched_to_iq_bytes(sched, code_bits);
    }
    if (total_steps < 2) {
        return sched_to_iq_bytes(sched, code_bits);
    }

    std::vector<SchedEntry> first_step_sched;
    std::vector<SchedEntry> remain_sched;
    first_step_sched.reserve(sched.size());
    remain_sched.reserve(sched.size());
    for (const auto& e : sched) {
        // step k 占用 cycle ∈ [k*step_span, (k+1)*step_span)；位于 (k+1)*step_span 的首事件属于下一步。
        if (e.abs_cycle < step_span) {
            first_step_sched.push_back(e);
        } else {
            remain_sched.push_back(e);
        }
    }
    if (first_step_sched.empty()) {
        return sched_to_iq_bytes(sched, code_bits);
    }
    if (remain_sched.empty()) {
        return sched_to_iq_bytes_no_end(first_step_sched, code_bits);
    }
    if (remain_sched.size() % (total_steps - 1) != 0) {
        return sched_to_iq_bytes(sched, code_bits);
    }
    // 剩余步以固定 step_span 为边界重定位，避免首步修正被后续归一化抵消。
    auto remain_normalized = normalize_sched_from_cycle(remain_sched, step_span);
    if (remain_normalized.empty()) {
        return sched_to_iq_bytes(sched, code_bits);
    }

    std::vector<uint8_t> data = sched_to_iq_bytes_no_end(first_step_sched, code_bits);
    // 首步与剩余 EMBED 段之间补齐 step 边界，避免后续控制块提前生效。
    const int32_t first_last_abs = static_cast<int32_t>(first_step_sched.back().abs_cycle);
    int32_t first_tail_extra = static_cast<int32_t>(step_span) - first_last_abs - 1;
    while (first_tail_extra > 0) {
        const uint32_t pad_dly = static_cast<uint32_t>(std::min<int32_t>(kMaxDlyField, first_tail_extra - 1));
        auto pad_inst = pack_instruction_legacy(1, 0, static_cast<uint8_t>(pad_dly), 0, code_bits);
        data.insert(data.end(), pad_inst.begin(), pad_inst.end());
        first_tail_extra -= static_cast<int32_t>(pad_dly) + 1;
    }
    auto remain_data = sched_to_iq_bytes_embed_loop(
        remain_normalized, code_bits, total_steps - 1, step_span, enable_inc
    );
    data.insert(data.end(), remain_data.begin(), remain_data.end());
    return data;
}

static uint32_t sum_issue_cycles_packed(const std::vector<uint8_t>& data,
                                        uint32_t width_bits,
                                        uint32_t code_bits) {
    const uint32_t inst_bytes = (width_bits + 7u) / 8u;
    if (inst_bytes == 0u || data.size() < inst_bytes) {
        return 0u;
    }
    const uint32_t n = static_cast<uint32_t>(data.size() / inst_bytes);
    if (n <= 1u) {
        return 0u;
    }
    const uint32_t total_bits = 41u + code_bits;
    uint32_t total = 0u;
    for (uint32_t i = 0; i + 1u < n; ++i) {
        unsigned __int128 inst = 0;
        const size_t off = static_cast<size_t>(i) * inst_bytes;
        for (uint32_t b = 0; b < inst_bytes && b < 16u; ++b) {
            inst = (inst << 8) | data[off + b];
        }
        const uint32_t dly =
            static_cast<uint32_t>((inst >> (total_bits - 31u)) & static_cast<unsigned __int128>(0x7u));
        total += dly + 1u;
    }
    return total;
}

/// 在 END 跳转前插入 vld=1 NOP，拉长该 IQ 的 issue 周期以与写回对端对齐。
static void pad_wb_iq_extra_issue_cycles(std::vector<uint8_t>& data,
                                         uint32_t code_bits,
                                         uint32_t extra_cycles) {
    if (extra_cycles == 0u || data.empty()) {
        return;
    }
    const uint32_t inst_bytes = (41u + code_bits + 7u) / 8u;
    if (data.size() < inst_bytes) {
        return;
    }
    auto insert_pos = data.end() - static_cast<std::ptrdiff_t>(inst_bytes);
    uint32_t remain = extra_cycles;
    while (remain > 0u) {
        const uint32_t dly = std::min<uint32_t>(remain > 0u ? remain - 1u : 0u, kMaxDlyField);
        auto nop = pack_instruction_legacy(1, 0, static_cast<uint8_t>(dly), 0, code_bits);
        if (nop.size() != inst_bytes) {
            break;
        }
        insert_pos = data.insert(insert_pos, nop.begin(), nop.end()) +
                     static_cast<std::ptrdiff_t>(inst_bytes);
        remain -= dly + 1u;
    }
}

static void pad_iq_before_end_jump(std::vector<uint8_t>& bytes,
                                   uint32_t width_bits,
                                   uint32_t target_inst_count,
                                   uint32_t code_bits) {
    const uint32_t inst_bytes = (width_bits + 7u) / 8u;
    if (inst_bytes == 0u || bytes.empty() || bytes.size() % inst_bytes != 0u)
        return;
    uint32_t cur_count = static_cast<uint32_t>(bytes.size() / inst_bytes);
    if (cur_count >= target_inst_count)
        return;
    if (cur_count < 1u)
        return;

    const auto nop = pack_instruction_legacy(1, 0, 0, 0, code_bits);
    if (nop.size() != inst_bytes)
        return;
    auto insert_pos = bytes.end() - static_cast<std::ptrdiff_t>(inst_bytes); // keep final jump as last
    while (cur_count < target_inst_count) {
        insert_pos = bytes.insert(insert_pos, nop.begin(), nop.end()) + static_cast<std::ptrdiff_t>(inst_bytes);
        ++cur_count;
    }
}

} // anonymous namespace

std::vector<IqInstructionFile> DivOpInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    auto calc_iq_depth = [](const std::vector<uint8_t>& data, uint32_t width_bits) -> uint32_t {
        const uint32_t inst_bytes = (width_bits + 7) / 8;
        return inst_bytes == 0 ? 0 : static_cast<uint32_t>(data.size() / inst_bytes);
    };

    auto add_iq = [&](const std::vector<SchedEntry>& sched,
                      uint32_t iq_id, const std::string& iq_name,
                      uint32_t code_bits,
                      bool use_embed_loop = false,
                      bool enforce_depth_limit = false,
                      bool embed_enable_inc = true) {
        if (sched.empty()) return;
        IqInstructionFile file;
        file.para_iq_id = iq_id;
        file.para_iq_name = iq_name;
        file.para_width_bits = 41 + code_bits;
        if (!use_embed_loop) {
            file.para_data = sched_to_iq_bytes_rtl_global_wall(sched, code_bits);
        } else {
            file.para_data = sched_to_iq_bytes_embed_loop(
                sched, code_bits, para_total_steps, para_step_span, embed_enable_inc
            );
        }
        const uint32_t iq_depth = calc_iq_depth(file.para_data, file.para_width_bits);
        if (enforce_depth_limit && iq_depth > kRtlIqDepth) {
            throw std::runtime_error(
                "Div IQ depth overflow: " + iq_name + " depth=" + std::to_string(iq_depth) +
                ", limit=" + std::to_string(kRtlIqDepth)
            );
        }
        files.push_back(file);
    };

    // UR‑RDATA / 写回 rtovr-35+ur-w0：EMBED 压缩（步长 CPI）。
    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 12, true, true, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 12, true, true, true);
    add_iq(sched_ur_r2, iq::IQ_ID_UR_RDATA_2, iq::IQ_NAME_UR_RDATA_2, 12, true, true, true);
    add_iq(sched_ur_r3, iq::IQ_ID_UR_RDATA_3, iq::IQ_NAME_UR_RDATA_3, 12, true, true, true);
    {
        validate_paired_rtovr35_ur_w(sched_rtovr_35, sched_ur_w0);

        // 单 step 写回：compile 固定 rv/ur 拍点 + EMBED 步长 CPI；多 step 仍走线性 abs 表。
        const bool use_fixed_wb_sched = (para_total_steps == 1u) &&
            validate_sched_for_mul_style_wb(
                sched_rtovr_35, sched_ur_w0, para_newton_iters, para_total_steps, para_step_span);
        const bool use_wb_embed =
            use_fixed_wb_sched && sched_wb_embed_eligible(sched_rtovr_35, para_newton_iters) &&
            sched_wb_embed_eligible(sched_ur_w0, para_newton_iters);

        uint8_t ur_inc_inner = 0u;
        if (para_total_steps > 1u && sched_ur_w0.size() >= 2u) {
            const uint32_t per_step = para_newton_iters + 1u;
            if (sched_ur_w0[per_step].code == (sched_ur_w0[0].code + (1u << 4)))
                ur_inc_inner = 1u;
        }

        // SMC：线性 + breaker（与 cypher 一致）；RTL：另存 *_rtl 成对 EMBED。
        std::vector<uint8_t> ur_wdata_bytes =
            pack_div_wb_rtl_linear(sched_ur_w0, 28, true, false);
        // #region agent log — hypothesis Q: rtovr breaker 与 ur 同拍导致 uvr_wdata 为零
        constexpr bool kRtlRtovrWbBreaker = true;
        // #endregion
        std::vector<uint8_t> rtovr_35_bytes =
            pack_div_wb_rtl_linear(sched_rtovr_35, 7, false, kRtlRtovrWbBreaker);

        bool ur_wb_same_output_addr = !sched_ur_w0.empty();
        if (ur_wb_same_output_addr) {
            const uint32_t ref_addr = (sched_ur_w0[0].code >> 4) & 0xFFu;
            const uint32_t ref_bank = (sched_ur_w0[0].code >> 1) & 0x7u;
            for (const auto& e : sched_ur_w0) {
                if (((e.code >> 4) & 0xFFu) != ref_addr ||
                    ((e.code >> 1) & 0x7u) != ref_bank) {
                    ur_wb_same_output_addr = false;
                    break;
                }
            }
        }

        const uint32_t wb_span = calc_div_wb_embed_span(sched_ur_w0, para_newton_iters);
        const bool ur_wb_embed_ready =
            sched_wb_embed_eligible(sched_ur_w0, para_newton_iters) &&
            sched_wb_embed_eligible(sched_rtovr_35, para_newton_iters);
        const bool embed_wall_ok_ur =
            wb_span > 0u &&
            ur_iq_rtl_emit::verify_wb_embed_iic(sched_ur_w0, para_newton_iters, wb_span, 28u);
        const bool embed_wall_ok_rv =
            wb_span > 0u &&
            ur_iq_rtl_emit::verify_wb_embed_iic(sched_rtovr_35, para_newton_iters, wb_span, 7u);
        const bool need_embed_for_rtl_addr =
            ur_wb_same_output_addr && sched_ur_w0.size() >= 2u;

        const uint32_t depth_linear = std::max(
            calc_iq_depth(ur_wdata_bytes, 41u + 28u),
            calc_iq_depth(rtovr_35_bytes, 41u + 7u));
        const bool need_embed_for_depth = depth_linear > kRtlIqDepth;

        if (need_embed_for_depth && use_wb_embed && embed_wall_ok_ur && embed_wall_ok_rv &&
                   ur_wb_embed_ready) {
            std::vector<uint8_t> embed_ur =
                pack_div_wb_embed_loop(sched_ur_w0, 28, para_newton_iters, ur_inc_inner != 0);
            std::vector<uint8_t> embed_rv =
                pack_div_wb_embed_loop(sched_rtovr_35, 7, para_newton_iters, false);
            const uint32_t depth_embed = std::max(
                calc_iq_depth(embed_ur, 41u + 28u), calc_iq_depth(embed_rv, 41u + 7u));
            if (depth_embed <= kRtlIqDepth) {
                ur_wdata_bytes = std::move(embed_ur);
                rtovr_35_bytes = std::move(embed_rv);
            }
        }

        auto push_padded = [&](std::vector<uint8_t>&& data, uint32_t iq_id, const std::string& iq_name,
                               uint32_t code_bits, bool enforce_depth_limit) {
            if (data.empty())
                return;
            IqInstructionFile file;
            file.para_iq_id = iq_id;
            file.para_iq_name = iq_name;
            file.para_width_bits = 41 + code_bits;
            file.para_data = std::move(data);
            const uint32_t iq_depth = calc_iq_depth(file.para_data, file.para_width_bits);
            if (enforce_depth_limit && iq_depth > kRtlIqDepth) {
                throw std::runtime_error(
                    "Div IQ depth overflow: " + iq_name + " depth=" + std::to_string(iq_depth) +
                    ", limit=" + std::to_string(kRtlIqDepth)
                );
            }
            files.push_back(std::move(file));
        };

        const uint32_t cyc_ur_smc = sum_issue_cycles_packed(ur_wdata_bytes, 41u + 28u, 28u);
        const uint32_t cyc_rv_smc = sum_issue_cycles_packed(rtovr_35_bytes, 41u + 7u, 7u);
        std::vector<uint8_t> ur_linear_for_rtl = ur_wdata_bytes;
        std::vector<uint8_t> rv_linear_for_rtl = rtovr_35_bytes;

        push_padded(std::move(ur_wdata_bytes), iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 28,
                    true);
        push_padded(std::move(rtovr_35_bytes), iq::IQ_ID_RTOVR_BASE, "rtovr-35", 7, true);

        bool rtl_embed_pushed = false;
        uint32_t cyc_ur_rtl = 0u;
        uint32_t cyc_rv_rtl = 0u;
        // RTL：ur-w0 EMBED+inc_embed（micro 槽）；rtovr-35 线性 global_wall；成对 issue 周期对齐。
        if (kEnableUrWbEmbedRtlAddr && need_embed_for_rtl_addr && ur_wb_embed_ready &&
            embed_wall_ok_ur) {
            std::vector<uint8_t> ur_rtl = pack_div_wb_embed_loop(
                sched_ur_w0, 28, para_newton_iters, ur_inc_inner != 0, true);
            std::vector<uint8_t> rv_rtl =
                pack_div_wb_rtl_linear(sched_rtovr_35, 7, false, true);
            if (!ur_rtl.empty() && !rv_rtl.empty()) {
                cyc_ur_rtl = sum_issue_cycles_packed(ur_rtl, 41u + 28u, 28u);
                cyc_rv_rtl = sum_issue_cycles_packed(rv_rtl, 41u + 7u, 7u);
                if (cyc_ur_rtl < cyc_rv_rtl) {
                    pad_wb_iq_extra_issue_cycles(ur_rtl, 28u, cyc_rv_rtl - cyc_ur_rtl);
                    cyc_ur_rtl = cyc_rv_rtl;
                } else if (cyc_rv_rtl < cyc_ur_rtl) {
                    pad_wb_iq_extra_issue_cycles(rv_rtl, 7u, cyc_ur_rtl - cyc_rv_rtl);
                    cyc_rv_rtl = cyc_ur_rtl;
                }
                push_padded(std::move(ur_rtl), iq::IQ_ID_UR_WDATA_0, "ur_wdata_0_rtl", 28, true);
                push_padded(std::move(rv_rtl), iq::IQ_ID_RTOVR_BASE, "rtovr-35_rtl", 7, true);
                rtl_embed_pushed = true;
                // #region agent log
                agent_debug_log(
                    "motion/div_op.cpp:get_iq_files",
                    "rtl_wb_pack_split",
                    "L",
                    "post-fix",
                    std::string("{\"ur_rtl_embed\":1,\"ur_rtl_breaker\":1,\"rv_rtl_linear\":1,") +
                        "\"rv_cycles\":" + std::to_string(cyc_rv_rtl) +
                        ",\"ur_cycles\":" + std::to_string(cyc_ur_rtl) + "}");
                // #endregion
            }
        } else {
            // 禁 EMBED 时仍写出与 SMC 相同的 *_rtl，避免 export 用到陈旧 EMBED 文件。
            push_padded(std::move(ur_linear_for_rtl), iq::IQ_ID_UR_WDATA_0, "ur_wdata_0_rtl", 28,
                        true);
            push_padded(std::move(rv_linear_for_rtl), iq::IQ_ID_RTOVR_BASE, "rtovr-35_rtl", 7,
                        true);
        }

        // #region agent log
        {
            const uint32_t wb_steps = para_newton_iters + 1u;
            const uint32_t body_len =
                wb_steps > 0u ? static_cast<uint32_t>(sched_ur_w0.size() / wb_steps) : 0u;
            agent_debug_log(
                "motion/div_op.cpp:get_iq_files",
                "wb_pack_gate",
                "B",
                "pre-fix",
                std::string("{\"wb_span\":") + std::to_string(wb_span) +
                    ",\"embed_wall_ok_ur\":" + std::to_string(embed_wall_ok_ur ? 1u : 0u) +
                    ",\"embed_wall_ok_rv\":" + std::to_string(embed_wall_ok_rv ? 1u : 0u) +
                    ",\"need_rtl_addr\":" + std::to_string(need_embed_for_rtl_addr ? 1u : 0u) +
                    ",\"rtl_embed_pushed\":" + std::to_string(rtl_embed_pushed ? 1u : 0u) +
                    ",\"k_enable_rtl_embed\":" + std::to_string(kEnableUrWbEmbedRtlAddr ? 1u : 0u) +
                    ",\"body_len\":" + std::to_string(body_len) +
                    ",\"ur_breaker_in_embed\":" +
                    std::to_string((body_len > 1u || kEnableUrWbEmbedRtlAddr) ? 1u : 0u) + "}");
            agent_debug_log(
                "motion/div_op.cpp:get_iq_files",
                "wb_issue_cycles",
                "A",
                "pre-fix",
                std::string("{\"cyc_ur_smc\":") + std::to_string(cyc_ur_smc) +
                    ",\"cyc_rv_smc\":" + std::to_string(cyc_rv_smc) +
                    ",\"cyc_ur_rtl\":" + std::to_string(cyc_ur_rtl) +
                    ",\"cyc_rv_rtl\":" + std::to_string(cyc_rv_rtl) + "}");
            if (!sched_ur_w0.empty()) {
                agent_debug_log(
                    "motion/div_op.cpp:get_iq_files",
                    "sched_wb_abs",
                    "D",
                    "pre-fix",
                    std::string("{\"ur0\":") + std::to_string(sched_ur_w0[0].abs_cycle) +
                        ",\"rv0\":" + std::to_string(sched_rtovr_35[0].abs_cycle) +
                        ",\"wb_ur_gap\":" + std::to_string(div::WB_UR_CYCLES_AFTER_RTOVR) +
                        (sched_ur_w0.size() > 1u
                             ? std::string(",\"ur1\":") + std::to_string(sched_ur_w0[1].abs_cycle) +
                                   ",\"rv1\":" + std::to_string(sched_rtovr_35[1].abs_cycle) +
                                   ",\"du\":" + std::to_string(sched_ur_w0[1].abs_cycle -
                                                               sched_ur_w0[0].abs_cycle)
                             : std::string("")) +
                        "}");
            }
        }
        // #endregion
    }
    const uint32_t ts = para_total_steps;
    const uint32_t ss = para_step_span;
    // 写回用 RTL 线性 + realign 墙钟；fpmul 强制线性，避免 step-EMBED 与写回墙钟错位。
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 3, false, true, false);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 5,
           sched_embed_loop_eligible(sched_fpadd, ts, ss), true, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_TO_FPMUL, "rtovr-3", 7,
           sched_embed_loop_eligible(sched_rtovr_3, ts, ss), true, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_TO_FPMUL, "rtovr-4", 7,
           sched_embed_loop_eligible(sched_rtovr_4, ts, ss), true, false);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_TO_FPADD, "rtovr-5", 7,
           sched_embed_loop_eligible(sched_rtovr_5, ts, ss), true, false);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_TO_FPADD, "rtovr-6", 7,
           sched_embed_loop_eligible(sched_rtovr_6, ts, ss), true, false);

    // #region agent log
    for (const auto& f : files) {
        const uint32_t code_bits = f.para_width_bits > 41u ? f.para_width_bits - 41u : 0u;
        const uint32_t cyc = sum_issue_cycles_packed(f.para_data, f.para_width_bits, code_bits);
        const uint32_t depth = calc_iq_depth(f.para_data, f.para_width_bits);
        if (f.para_iq_name == "fpmul" || f.para_iq_name == iq::IQ_NAME_UR_RDATA_0 ||
            f.para_iq_name == iq::IQ_NAME_UR_WDATA_0 || f.para_iq_name == "rtovr-35" ||
            f.para_iq_name == "ur_wdata_0_rtl" || f.para_iq_name == "rtovr-35_rtl") {
            agent_debug_log(
                "motion/div_op.cpp:get_iq_files",
                "iq_wall_cycles",
                "A",
                "pre-fix",
                std::string("{\"iq\":\"") + f.para_iq_name + "\",\"cycles\":" +
                    std::to_string(cyc) + ",\"depth\":" + std::to_string(depth) + "}");
        }
    }
    // #endregion

    return files;
}

std::vector<uint8_t> DivOpInstructions::serialize() const {
    std::vector<uint8_t> result;
    auto push_u32 = [&](uint32_t v) {
        result.push_back(v & 0xFF);
        result.push_back((v >> 8) & 0xFF);
        result.push_back((v >> 16) & 0xFF);
        result.push_back((v >> 24) & 0xFF);
    };
    push_u32(para_total_steps);
    push_u32(para_total_elements);
    push_u32(para_newton_iters);
    push_u32(para_total_cycles);
    push_u32(para_step_span);
    result.push_back(para_fpmul_code);
    result.push_back(para_fpadd_code);

    auto ser_sched = [&](const std::vector<SchedEntry>& s) {
        push_u32((uint32_t)s.size());
        for (const auto& e : s) { push_u32(e.abs_cycle); push_u32(e.code); }
    };
    ser_sched(sched_ur_r0);  ser_sched(sched_ur_r1);  ser_sched(sched_ur_r2); ser_sched(sched_ur_r3);
    ser_sched(sched_ur_w0);  ser_sched(sched_fpmul);   ser_sched(sched_fpadd);
    ser_sched(sched_rtovr_3); ser_sched(sched_rtovr_4);
    ser_sched(sched_rtovr_5); ser_sched(sched_rtovr_6); ser_sched(sched_rtovr_35);
    return result;
}

bool DivOp::calculate_golden(const app::AppOperatorRequest& req,
                              std::vector<uint8_t>& golden_bin) {
    golden_bin.clear();
    const auto& inputs = req.para_inputs;
    std::vector<float> N_f32, D_f32, X0_f32;
    app::AppDataType dtype = app::AppDataType::FLOAT32;

    if (inputs.find("input_tensor") != inputs.end()) {
        const auto& t = inputs.at("input_tensor");
        dtype = t.para_dtype;
        N_f32 = t.para_data_f32;
    }
    if (inputs.find("other_tensor") != inputs.end())
        D_f32 = inputs.at("other_tensor").para_data_f32;
    if (D_f32.empty()) {
        return false;
    }

    auto alpha_it = req.para_parameters.find("alpha");
    if (alpha_it != req.para_parameters.end())
        X0_f32 = alpha_it->second.para_data_f32;

    auto fast_rcp_x0 = [](float d) -> float {
        uint32_t i = 0; std::memcpy(&i, &d, sizeof(float));
        i = 0x7EF311C7u - i;
        float x; std::memcpy(&x, &i, sizeof(float)); return x;
    };

    int newton_iters = 3;
    auto iters_it = req.para_parameters.find("iters");
    if (iters_it != req.para_parameters.end() && !iters_it->second.para_data_i32.empty())
        newton_iters = iters_it->second.para_data_i32[0];
    auto iters_it2 = req.para_parameters.find("newton_iters");
    if (iters_it2 != req.para_parameters.end() && !iters_it2->second.para_data_i32.empty())
        newton_iters = iters_it2->second.para_data_i32[0];
    if (newton_iters > 4) newton_iters = 4;
    {
        const uint32_t epw = (dtype == app::AppDataType::FLOAT16) ? 8u : 4u;
        const uint32_t steps =
            static_cast<uint32_t>((static_cast<uint64_t>(N_f32.size()) + epw - 1) / epw);
        newton_iters = static_cast<int>(
            div::clamp_div_newton_iters_for_rtl_iq(steps, static_cast<uint32_t>(newton_iters)));
    }

    // 与 compile() 内 get_iq_files() 深度重试对齐，保证金标 Newton 次数与微码一致。
    {
        DivOpParams sync_p;
        sync_p.para_n = req.para_dimensions.para_n;
        sync_p.para_c = req.para_dimensions.para_c;
        sync_p.para_h = req.para_dimensions.para_h;
        sync_p.para_w = req.para_dimensions.para_w;
        sync_p.para_dtype = (dtype == app::AppDataType::FLOAT16) ? 1u : 0u;
        sync_p.para_iters = static_cast<uint32_t>(newton_iters);
        sync_p.para_N_bank = read_bank_param(req, "N_bank", 0);
        sync_p.para_D_bank = read_bank_param(req, "D_bank", 1);
        sync_p.para_x0_bank = read_bank_param(req, "x0_bank", 2);
        sync_p.para_const2_bank = read_bank_param(req, "const2_bank", 3);
        sync_p.para_output_bank = read_bank_param(req, "output_bank", 4);
        sync_p.para_N_addr = read_u32_addr_param(req, "N_addr", 0);
        sync_p.para_D_addr = read_u32_addr_param(req, "D_addr", 0);
        sync_p.para_x0_addr = read_u32_addr_param(req, "x0_addr", 0);
        sync_p.para_const2_addr = read_u32_addr_param(req, "const2_addr", 0);
        sync_p.para_output_addr = read_u32_addr_param(req, "output_addr", 0);
        DivOp dg;
        (void)dg.compile(sync_p);
        newton_iters = static_cast<int>(sync_p.para_iters);
    }

    auto fp32_force = [](float v) -> float {
        volatile float tmp = v;
        return tmp;
    };

    auto fp16_trunc = [](float f) -> uint16_t {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        uint32_t sign = (bits >> 31) & 0x1;
        uint32_t exp  = (bits >> 23) & 0xFF;
        uint32_t mant = bits & 0x7FFFFF;
        if (exp == 0) return static_cast<uint16_t>(sign << 15);
        if (exp == 0xFF) return static_cast<uint16_t>((sign << 15) | (0x1F << 10) | (mant >> 13));
        int32_t fp16_exp = static_cast<int32_t>(exp) - 127 + 15;
        if (fp16_exp <= 0) return static_cast<uint16_t>(sign << 15);
        if (fp16_exp >= 0x1F) return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        uint16_t fp16_mant = static_cast<uint16_t>(mant >> 13);
        return static_cast<uint16_t>((sign << 15) | (fp16_exp << 10) | fp16_mant);
    };

    for (size_t idx = 0; idx < N_f32.size(); ++idx) {
        float n_val = N_f32[idx];
        float d_val = D_f32[idx];
        float x = (!X0_f32.empty() && idx < X0_f32.size()) ? X0_f32[idx] : fast_rcp_x0(d_val);

        if (dtype == app::AppDataType::FLOAT16) {
            uint16_t n16 = app::float32_to_float16(n_val);
            uint16_t d16 = app::float32_to_float16(d_val);
            float n_t = app::float16_to_float32(n16);
            float d_t = app::float16_to_float32(d16);
            float x_t = app::float16_to_float32(app::float32_to_float16(x));

            for (int n = 0; n < newton_iters; ++n) {
                float e   = app::float16_to_float32(app::float32_to_float16(d_t * x_t));
                float sub = app::float16_to_float32(fp16_trunc(2.0f - e));
                x_t       = app::float16_to_float32(app::float32_to_float16(x_t * sub));
            }

            float result = app::float16_to_float32(app::float32_to_float16(n_t * x_t));
            uint16_t bits = app::float32_to_float16(result);
            golden_bin.push_back(bits & 0xFF);
            golden_bin.push_back((bits >> 8) & 0xFF);
        } else {
            for (int n = 0; n < newton_iters; ++n) {
                float e   = fp32_force(d_val * x);
                float sub = fp32_force(2.0f - e);
                x         = fp32_force(x * sub);
            }

            float result = fp32_force(n_val * x);
            uint32_t bits; memcpy(&bits, &result, sizeof(float));
            golden_bin.push_back(bits & 0xFF);
            golden_bin.push_back((bits >> 8) & 0xFF);
            golden_bin.push_back((bits >> 16) & 0xFF);
            golden_bin.push_back((bits >> 24) & 0xFF);
        }
    }
    return true;
}

void DivOp::fill_random_request(app::AppOperatorRequest& req,
                                std::mt19937& rng,
                                std::uniform_real_distribution<float>& data_dist,
                                const std::string& dtype_str) {
    req.para_operator = "Div";
    uint32_t elems_per_word = (dtype_str == "fp16") ? 8u : 4u;

    std::uniform_int_distribution<int> iters_dist(1, 4);
    int rand_iters = iters_dist(rng);
    // ur-w0 / rtovr-35 仍为线性：大尺寸下须收缩体积或迭代，否则 INSTQ 深度 >32。
    for (int guard = 0; guard < 48; ++guard) {
        uint32_t total_now = req.para_dimensions.para_n * req.para_dimensions.para_c *
                             req.para_dimensions.para_h * req.para_dimensions.para_w;
        const uint32_t steps = (total_now + elems_per_word - 1) / elems_per_word;
        rand_iters = static_cast<int>(div::clamp_div_newton_iters_for_rtl_iq(
            steps, static_cast<uint32_t>(rand_iters)));
        if (div::div_est_ur_pair_linear_inst_slots(steps, static_cast<uint32_t>(rand_iters)) <=
            static_cast<uint64_t>(div::RTL_IQ_DEPTH_LIMIT))
            break;
        auto& d = req.para_dimensions;
        if (d.para_w > 1u)
            d.para_w = std::max(1u, d.para_w / 2u);
        else if (d.para_h > 1u)
            d.para_h = std::max(1u, d.para_h / 2u);
        else if (d.para_c > 1u)
            d.para_c = std::max(1u, d.para_c / 2u);
        else if (d.para_n > 1u)
            d.para_n = std::max(1u, d.para_n / 2u);
        else {
            // 无法再缩维度时强制最小体积，避免 linear ur-w0 条数爆炸。
            d.para_n = d.para_c = d.para_h = d.para_w = 1u;
            rand_iters = 1;
            break;
        }
    }

    uint32_t total = req.para_dimensions.para_n * req.para_dimensions.para_c *
                     req.para_dimensions.para_h * req.para_dimensions.para_w;
    app::AppTensorDesc td;
    td.para_name = "newton_iters";
    td.para_dtype = app::AppDataType::INT32;
    td.para_shape = {1};
    td.para_data_i32.push_back(rand_iters);
    req.para_parameters["newton_iters"] = td;
    req.para_parameters.erase("iters");

    app::AppDataType target_type = (dtype_str == "fp16")
                                 ? app::AppDataType::FLOAT16 : app::AppDataType::FLOAT32;

    app::AppTensorDesc input_desc;
    input_desc.para_name = "input_tensor";
    input_desc.para_dtype = target_type;
    input_desc.para_shape = {req.para_dimensions.para_n, req.para_dimensions.para_c,
                             req.para_dimensions.para_h, req.para_dimensions.para_w};
    for (uint32_t i = 0; i < total; ++i)
        input_desc.para_data_f32.push_back(data_dist(rng));
    req.para_inputs["input_tensor"] = input_desc;

    app::AppTensorDesc other_desc;
    other_desc.para_name = "other_tensor";
    other_desc.para_dtype = target_type;
    other_desc.para_shape = input_desc.para_shape;
    std::uniform_real_distribution<float> denom_dist(0.5f, 5.0f);

    app::AppTensorDesc alpha_desc;
    alpha_desc.para_name = "alpha";
    alpha_desc.para_dtype = target_type;
    alpha_desc.para_shape = input_desc.para_shape;

    for (uint32_t i = 0; i < total; ++i) {
        float d = denom_dist(rng);
        other_desc.para_data_f32.push_back(d);
        uint32_t bits; std::memcpy(&bits, &d, sizeof(float));
        bits = 0x7EF311C7u - bits;
        float x0; std::memcpy(&x0, &bits, sizeof(float));
        alpha_desc.para_data_f32.push_back(x0);
    }
    req.para_inputs["other_tensor"] = other_desc;
    req.para_parameters["alpha"] = alpha_desc;

    auto add_int_param = [&](const std::string& name, int val) {
        app::AppTensorDesc td;
        td.para_name = name;
        td.para_dtype = app::AppDataType::INT32;
        td.para_shape = {1};
        td.para_data_i32.push_back(val);
        req.para_parameters[name] = td;
    };

    app::AppTensorDesc const_desc;
    const_desc.para_name = "const_tensor";
    const_desc.para_dtype = target_type;
    uint32_t const_lanes = elems_per_word;
    const_desc.para_shape = {1, 1, 1, const_lanes};
    for (uint32_t i = 0; i < const_lanes; ++i) const_desc.para_data_f32.push_back(2.0f);
    req.para_parameters["const_tensor"] = const_desc;

    // 本版本忽略 bank_conflict：固定无冲突 bank 与地址映射。
    req.para_layout_mode = "default";
    req.para_bc_input_bank = -1;
    req.para_bc_other_bank = -1;
    req.para_bc_alpha_bank = -1;
    req.para_bc_output_bank = -1;

    const int N_bank = 0;
    const int D_bank = 1;
    const int x0_bank = 2;
    const int const2_bank = 3;
    const int output_bank = 4;
    const int N_addr = 0;
    const int D_addr = 0;
    const int x0_addr = 0;
    const int const2_addr = 0;
    const int output_addr = 0;

    add_int_param("input_bank", N_bank);
    add_int_param("other_bank", D_bank);
    add_int_param("alpha_bank", x0_bank);
    add_int_param("const2_bank", const2_bank);
    add_int_param("output_bank", output_bank);
    add_int_param("input_addr", N_addr);
    add_int_param("other_addr", D_addr);
    add_int_param("alpha_addr", x0_addr);
    add_int_param("const2_addr", const2_addr);
    add_int_param("output_addr", output_addr);

    add_int_param("N_bank", N_bank);
    add_int_param("N_addr", N_addr);
    add_int_param("D_bank", D_bank);
    add_int_param("D_addr", D_addr);
    add_int_param("x0_bank", x0_bank);
    add_int_param("x0_addr", x0_addr);
}

bool DivOp::generate_data(const app::AppOperatorRequest& req,
                           std::vector<uint8_t>& input_bin,
                           std::vector<uint8_t>& other_bin,
                           std::vector<uint8_t>& alpha_bin) {
    input_bin.clear(); other_bin.clear(); alpha_bin.clear();

    auto write_f32 = [](std::vector<uint8_t>& bin, const std::vector<float>& data) {
        for (float v : data) {
            uint32_t bits; memcpy(&bits, &v, sizeof(float));
            bin.push_back(bits & 0xFF);       bin.push_back((bits >> 8) & 0xFF);
            bin.push_back((bits >> 16) & 0xFF); bin.push_back((bits >> 24) & 0xFF);
        }
    };
    auto write_f16 = [](std::vector<uint8_t>& bin, const std::vector<float>& data) {
        for (float v : data) {
            uint16_t bits = app::float32_to_float16(v);
            bin.push_back(bits & 0xFF);
            bin.push_back((bits >> 8) & 0xFF);
        }
    };
    auto write_td = [&](std::vector<uint8_t>& bin, const app::AppTensorDesc& t) {
        if (t.para_dtype == app::AppDataType::FLOAT32) write_f32(bin, t.para_data_f32);
        else                                            write_f16(bin, t.para_data_f32);
    };

    const auto& inputs = req.para_inputs;
    if (inputs.find("input_tensor") != inputs.end()) write_td(input_bin, inputs.at("input_tensor"));
    if (inputs.find("other_tensor") != inputs.end()) write_td(other_bin, inputs.at("other_tensor"));
    const auto& params = req.para_parameters;
    if (params.find("alpha") != params.end()) write_td(alpha_bin, params.at("alpha"));

    return true;
}

void print_div_instructions(const DivOpInstructions& insts) {
    std::cout << "=== Div Op Instructions (schedule-based, 15 CPI + 7 final) ===\n"
              << "  out[i] = input[i] / other[i]\n\n"
              << "Basic Info:\n"
              << "  Total steps: " << insts.para_total_steps << "\n"
              << "  Total elements: " << insts.para_total_elements << "\n"
              << "  Newton iters: " << insts.para_newton_iters << "\n"
              << "  Total cycles: " << insts.para_total_cycles << "\n";
}

REGISTER_OP(DivOp);

} // namespace op
} // namespace aiacc
