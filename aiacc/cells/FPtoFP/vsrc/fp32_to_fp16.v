//----------------------------------------------------------------------------
// Filename: fp32_to_fp16.v
// Author: xiaoke
// Date: 2025-10-24
// Version: 1.0
// Description: FP32 to FP16 conversion unit
//              Converts 32-bit single-precision to 16-bit half-precision
//----------------------------------------------------------------------------

module fp32_to_fp16 (
    input  wire [31:0] fp32_in,
    output wire [15:0] fp16_out
);

    // Extract FP32 fields
    wire        sign_32 = fp32_in[31];
    wire [7:0]  exp_32  = fp32_in[30:23];
    wire [22:0] mant_32 = fp32_in[22:0];
    
    // Special value detection
    wire is_zero_32 = (exp_32 == 8'd0) && (mant_32 == 23'd0);
    wire is_inf_32   = (exp_32 == 8'd255) && (mant_32 == 23'd0);
    wire is_nan_32   = (exp_32 == 8'd255) && (mant_32 != 23'd0);
    
    // FP16 fields
    wire        sign_16;
    wire [4:0]  exp_16;
    wire [9:0]  mant_16;
    
    // Sign bit (same for both)
    assign sign_16 = sign_32;
    
    // Exponent conversion with overflow/underflow handling
    wire [8:0] exp_diff = {1'b0, exp_32} - 9'd127 + 9'd15; // bias adjustment
    
    assign exp_16 = is_zero_32 ? 5'd0 :
                    is_inf_32 || is_nan_32 ? 5'd31 :
                    (exp_diff[8]) ? 5'd0 :  // Underflow to zero
                    (exp_diff > 9'd31) ? 5'd31 :  // Overflow to infinity
                    exp_diff[4:0];
    
    // Mantissa conversion with rounding
    assign mant_16 = is_zero_32 ? 10'd0 :
                     is_inf_32 || is_nan_32 ? mant_32[22:13] :
                     (exp_diff[8]) ? 10'd0 :  // Underflow
                     (exp_diff > 9'd31) ? 10'd0 :  // Overflow
                     mant_32[22:13];  // Truncate to 10 bits
    
    // Output assembly
    assign fp16_out = {sign_16, exp_16, mant_16};

endmodule
