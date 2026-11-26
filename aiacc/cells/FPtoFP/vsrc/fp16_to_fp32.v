//----------------------------------------------------------------------------
// Filename: fp16_to_fp32.v
// Author: xiaoke
// Date: 2025-10-24
// Version: 1.0
// Description: FP16 to FP32 conversion unit
//              Converts 16-bit half-precision to 32-bit single-precision
//----------------------------------------------------------------------------

module fp16_to_fp32 (
    input  wire [15:0] fp16_in,
    output wire [31:0] fp32_out
);

    // Extract FP16 fields
    wire        sign_16 = fp16_in[15];
    wire [4:0]  exp_16  = fp16_in[14:10];
    wire [9:0]  mant_16 = fp16_in[9:0];
    
    // Special value detection
    wire is_zero_16 = (exp_16 == 5'd0) && (mant_16 == 10'd0);
    wire is_inf_16  = (exp_16 == 5'd31) && (mant_16 == 10'd0);
    wire is_nan_16  = (exp_16 == 5'd31) && (mant_16 != 10'd0);
    
    // FP32 fields
    wire        sign_32;
    wire [7:0]  exp_32;
    wire [22:0] mant_32;
    
    // Sign bit (same for both)
    assign sign_32 = sign_16;
    
    // Exponent conversion
    assign exp_32 = is_zero_16 ? 8'd0 :
                    is_inf_16 || is_nan_16 ? 8'd255 :
                    (exp_16 == 5'd0) ? 8'd0 :  // Denormalized
                    (exp_16 - 5'd15) + 8'd127; // Normalized: bias adjustment
    
    // Mantissa conversion
    assign mant_32 = is_zero_16 ? 23'd0 :
                     is_inf_16 || is_nan_16 ? {mant_16, 13'd0} :
                     (exp_16 == 5'd0) ? {mant_16, 13'd0} :  // Denormalized
                     {mant_16, 13'd0};  // Normalized
    
    // Output assembly
    assign fp32_out = {sign_32, exp_32, mant_32};

endmodule
