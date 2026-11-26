#ifndef DIV_OP_COMPUTE_INST_H
#define DIV_OP_COMPUTE_INST_H

/**
 * @file div_op_compute_inst.h
 * @brief Div 计算单元微指令生成（FPMUL, FPADD_SUB）
 */

#include <cstdint>
#include "../../iq/modules/fpmul_iq.h"
#include "../../iq/modules/fpadd_iq.h"

namespace aiacc {
namespace op {

struct DivOpParams;

void div_gen_fpmul_inst(const DivOpParams& params, iq::FpMulCru& cru);
void div_gen_fpadd_sub_inst(const DivOpParams& params, iq::FpAddCru& cru);

} // namespace op
} // namespace aiacc

#endif // DIV_OP_COMPUTE_INST_H