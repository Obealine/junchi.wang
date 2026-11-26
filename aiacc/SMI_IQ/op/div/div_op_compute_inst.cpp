#include "div_op_compute_inst.h"
#include "div_op.h"

namespace aiacc {
namespace op {

void div_gen_fpmul_inst(const DivOpParams& params, iq::FpMulCru& cru) {
    cru.para_inst_valid = true;
    cru.para_src_precision_is_32b = (params.para_dtype == 0);
    cru.para_dst_precision_is_32b = (params.para_dtype == 0);
}

void div_gen_fpadd_sub_inst(const DivOpParams& params, iq::FpAddCru& cru) {
    cru.para_inst_valid = true;
    cru.para_mode_is_32b = (params.para_dtype == 0);
    cru.para_update_st = false;
    cru.para_is_sub = true;
}

} // namespace op
} // namespace aiacc