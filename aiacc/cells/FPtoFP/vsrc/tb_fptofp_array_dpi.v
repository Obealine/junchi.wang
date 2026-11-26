//-----------------------------------------------------------------------------
// Filename: tb_fptofp_array_dpi.v
// Author: xiaoke
// Date: 2025-11-30
// Version: 1.0
// Description: Verify the functional correctness of fptofp_array
//              using DPI-C reference model
//-----------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_fptofp_array_dpi;

    parameter CLK_PERIOD        = 10;
    parameter RESET_CYCLES      = 10;
    parameter NUM_RANDOM_TESTS  = 100000;
    parameter RANDOM_SEED       = 42;
    parameter PRINT_ONLY_FAILURES = 1; // 1: 只打印失败案例, 0: 打印所有案例

    // DPI函数声明
    import "DPI-C" function int unsigned fptofp_fp16_to_fp32(
        input shortint unsigned fp16_val
    );
    import "DPI-C" function int unsigned fptofp_fp32_to_fp16(
        input int unsigned fp32_val
    );
    import "DPI-C" function int unsigned fptofp_fp16_to_fp16(
        input shortint unsigned fp16_val
    );
    import "DPI-C" function int unsigned fptofp_fp32_to_fp32(
        input int unsigned fp32_val
    );

    // 时钟和复位
    reg clk, rst_n;
    
    // 输入数据寄存器
    reg [127:0] dvr_fptofp_s;
    reg [4:0]   cru_fptofp;
    reg [4:0]   smc_id;
    
    // 输出数据线
    wire [127:0] dr_fptofp_d;
    
    // 测试变量
    integer   test_count, pass_count, fail_count;
    reg [15:0] test_inputs_fp16 [0:7];   // FP16输入（8路）
    reg [31:0] test_inputs_fp32 [0:3];  // FP32输入（4路）
    reg [15:0] expected_result_fp16 [0:7];
    reg [31:0] expected_result_fp32 [0:3];
    reg [15:0] actual_result_fp16 [0:7];
    reg [31:0] actual_result_fp32 [0:3];
    reg [1:0]  conversion_mode;  // 0:FP16toFP16, 1:FP32toFP32, 2:FP16toFP32, 3:FP32toFP16
    
    // 中间变量监控
    wire [4:0]  debug_cru_r;
    wire        debug_instruction_valid;
    wire        debug_src_precision;
    wire        debug_dest_precision;
    wire        debug_mode_fp32_to_fp32;
    wire        debug_mode_fp16_to_fp16;
    wire        debug_mode_fp16_to_fp32;
    wire        debug_mode_fp32_to_fp16;
    wire [127:0] debug_output_data;
    wire [127:0] debug_output_data_r;
    
    // 连接到DUT的内部信号（通过层次化访问）
    assign debug_cru_r = dut.cru_r;
    assign debug_instruction_valid = dut.instruction_valid;
    assign debug_src_precision = dut.src_precision;
    assign debug_dest_precision = dut.dest_precision;
    assign debug_mode_fp32_to_fp32 = dut.mode_fp32_to_fp32;
    assign debug_mode_fp16_to_fp16 = dut.mode_fp16_to_fp16;
    assign debug_mode_fp16_to_fp32 = dut.mode_fp16_to_fp32;
    assign debug_mode_fp32_to_fp16 = dut.mode_fp32_to_fp16;
    assign debug_output_data = dut.output_data;
    assign debug_output_data_r = dut.output_data_r;
    
    integer log_file;

    // 实例化被测设计
    fptofp_array dut (
        .clk(clk),
        .rst_n(rst_n),
        .dvr_fptofp_s(dvr_fptofp_s),
        .cru_fptofp(cru_fptofp),
        .dr_fptofp_d(dr_fptofp_d),
        .smc_id(smc_id)
    );

    // 时钟生成
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // 复位任务
    task reset_dut;
        begin
            rst_n = 1'b0;
            cru_fptofp = 5'b0;
            dvr_fptofp_s = 128'b0;
            smc_id = 5'd0;
            #(CLK_PERIOD);
            rst_n = 1'b1;
        end
    endtask

    // 计算并加载寄存器任务（FP16toFP16模式）
    task calculate_and_load_regs_fp16tofp16;
        input [15:0] new_inputs [0:7];
        integer i;
    begin
        dvr_fptofp_s = {
            new_inputs[7], new_inputs[6], new_inputs[5], new_inputs[4],
            new_inputs[3], new_inputs[2], new_inputs[1], new_inputs[0]
        };
        cru_fptofp = 5'b00001; // [0]=valid=1, [1]=src_precision=0, [2]=dest_precision=0, [3]=src_pos=0, [4]=dest_pos=0
    end
    endtask

    // 计算并加载寄存器任务（FP32toFP32模式）
    task calculate_and_load_regs_fp32tofp32;
        input [31:0] new_inputs [0:3];
        integer i;
    begin
        dvr_fptofp_s = {
            new_inputs[3], new_inputs[2], new_inputs[1], new_inputs[0]
        };
        cru_fptofp = 5'b00111; // [0]=valid=1, [1]=src_precision=1, [2]=dest_precision=1, [3]=src_pos=0, [4]=dest_pos=0
    end
    endtask

    // 计算并加载寄存器任务（FP16toFP32模式）
    task calculate_and_load_regs_fp16tofp32;
        input [15:0] new_inputs [0:7];
        input        src_pos_in;
        integer i;
    begin
        dvr_fptofp_s = {
            new_inputs[7], new_inputs[6], new_inputs[5], new_inputs[4],
            new_inputs[3], new_inputs[2], new_inputs[1], new_inputs[0]
        };
        cru_fptofp = {1'b0, src_pos_in, 1'b1, 1'b0, 1'b1}; // [4]=dest_pos, [3]=src_pos, [2]=dest_precision, [1]=src_precision, [0]=valid
    end
    endtask

    // 计算并加载寄存器任务（FP32toFP16模式）
    task calculate_and_load_regs_fp32tofp16;
        input [31:0] new_inputs [0:3];
        input        dest_pos_in;
        integer i;
    begin
        dvr_fptofp_s = {
            new_inputs[3], new_inputs[2], new_inputs[1], new_inputs[0]
        };
        cru_fptofp = {dest_pos_in, 1'b0, 1'b0, 1'b1, 1'b1}; // [4]=dest_pos, [3]=src_pos, [2]=dest_precision, [1]=src_precision, [0]=valid
    end
    endtask

    // 计算期望值任务（FP16toFP16）
    task compute_expected_values_fp16tofp16;
        integer i;
    begin
        for (i = 0; i < 8; i = i + 1) begin
            expected_result_fp16[i] = fptofp_fp16_to_fp16(test_inputs_fp16[i]);
        end
    end
    endtask

    // 计算期望值任务（FP32toFP32）
    task compute_expected_values_fp32tofp32;
        integer i;
    begin
        for (i = 0; i < 4; i = i + 1) begin
            expected_result_fp32[i] = fptofp_fp32_to_fp32(test_inputs_fp32[i]);
        end
    end
    endtask

    // 计算期望值任务（FP16toFP32）
    task compute_expected_values_fp16tofp32;
        integer i;
    begin
        for (i = 0; i < 4; i = i + 1) begin
            // 根据src_pos选择输入
            if (cru_fptofp[3]) begin
                expected_result_fp32[i] = fptofp_fp16_to_fp32(test_inputs_fp16[2*i+1]);
            end else begin
                expected_result_fp32[i] = fptofp_fp16_to_fp32(test_inputs_fp16[2*i]);
            end
        end
    end
    endtask

    // 计算期望值任务（FP32toFP16）
    task compute_expected_values_fp32tofp16;
        integer i;
    begin
        for (i = 0; i < 4; i = i + 1) begin
            expected_result_fp16[2*i] = fptofp_fp32_to_fp16(test_inputs_fp32[i]);
            expected_result_fp16[2*i+1] = expected_result_fp16[2*i]; // 复制到两个位置
        end
    end
    endtask

    // 从输出寄存器提取实际结果（FP16toFP16）
    task extract_actual_results_fp16tofp16;
        integer i;
    begin
        for (i = 0; i < 8; i = i + 1) begin
            actual_result_fp16[i] = dr_fptofp_d[16*i +: 16];
        end
    end
    endtask

    // 从输出寄存器提取实际结果（FP32toFP32）
    task extract_actual_results_fp32tofp32;
        integer i;
    begin
        for (i = 0; i < 4; i = i + 1) begin
            actual_result_fp32[i] = dr_fptofp_d[32*i +: 32];
        end
    end
    endtask

    // 从输出寄存器提取实际结果（FP16toFP32）
    task extract_actual_results_fp16tofp32;
        integer i;
    begin
        for (i = 0; i < 4; i = i + 1) begin
            actual_result_fp32[i] = dr_fptofp_d[32*i +: 32];
        end
    end
    endtask

    // 从输出寄存器提取实际结果（FP32toFP16）
    task extract_actual_results_fp32tofp16;
        integer i;
    begin
        for (i = 0; i < 8; i = i + 1) begin
            actual_result_fp16[i] = dr_fptofp_d[16*i +: 16];
        end
    end
    endtask

    // 检查结果任务（增强版：添加详细错误输出）
    task check_results;
        input  [1023:0] test_name;
        input  integer  test_num;
        output reg      passed;
        integer i;
        reg lane_passed;
        integer error_count;
    begin
        passed = 1'b1;
        error_count = 0;
        
        $fdisplay(log_file, "\n========================================");
        $fdisplay(log_file, "Test %0d: %0s", test_num, test_name);
        $fdisplay(log_file, "Conversion Mode: %0d", conversion_mode);
        $fdisplay(log_file, "========================================");
        
        case (conversion_mode)
            2'd0: begin // FP16toFP16
                for (i = 0; i < 8; i = i + 1) begin
                    lane_passed = (actual_result_fp16[i] == expected_result_fp16[i]);
                    if (!lane_passed) begin
                        $fdisplay(log_file, "\n  >>> ERROR: Lane %0d MISMATCH <<<", i);
                        $fdisplay(log_file, "    Input FP16:  0x%h", test_inputs_fp16[i]);
                        $fdisplay(log_file, "    Actual FP16:  0x%h", actual_result_fp16[i]);
                        $fdisplay(log_file, "    Expected FP16: 0x%h", expected_result_fp16[i]);
                        $fdisplay(log_file, "    Difference:    0x%h", actual_result_fp16[i] ^ expected_result_fp16[i]);
                        error_count = error_count + 1;
                        passed = 1'b0;
                    end
                end
            end
            2'd1: begin // FP32toFP32
                for (i = 0; i < 4; i = i + 1) begin
                    lane_passed = (actual_result_fp32[i] == expected_result_fp32[i]);
                    if (!lane_passed) begin
                        $fdisplay(log_file, "\n  >>> ERROR: Lane %0d MISMATCH <<<", i);
                        $fdisplay(log_file, "    Input FP32:  0x%h", test_inputs_fp32[i]);
                        $fdisplay(log_file, "    Actual FP32:  0x%h", actual_result_fp32[i]);
                        $fdisplay(log_file, "    Expected FP32: 0x%h", expected_result_fp32[i]);
                        $fdisplay(log_file, "    Difference:    0x%h", actual_result_fp32[i] ^ expected_result_fp32[i]);
                        error_count = error_count + 1;
                        passed = 1'b0;
                    end
                end
            end
            2'd2: begin // FP16toFP32
                for (i = 0; i < 4; i = i + 1) begin
                    lane_passed = (actual_result_fp32[i] == expected_result_fp32[i]);
                    if (!lane_passed) begin
                        $fdisplay(log_file, "\n  >>> ERROR: Lane %0d MISMATCH <<<", i);
                        $fdisplay(log_file, "    Input FP16:  0x%h", test_inputs_fp16[2*i + cru_fptofp[3]]);
                        $fdisplay(log_file, "    Actual FP32:  0x%h", actual_result_fp32[i]);
                        $fdisplay(log_file, "    Expected FP32: 0x%h", expected_result_fp32[i]);
                        $fdisplay(log_file, "    Difference:    0x%h", actual_result_fp32[i] ^ expected_result_fp32[i]);
                        error_count = error_count + 1;
                        passed = 1'b0;
                    end
                end
            end
            2'd3: begin // FP32toFP16
                for (i = 0; i < 4; i = i + 1) begin
                    lane_passed = (actual_result_fp16[2*i] == expected_result_fp16[2*i]) &&
                                  (actual_result_fp16[2*i+1] == expected_result_fp16[2*i+1]);
                    if (!lane_passed) begin
                        $fdisplay(log_file, "\n  >>> ERROR: Lane %0d MISMATCH <<<", i);
                        $fdisplay(log_file, "    Input FP32:  0x%h", test_inputs_fp32[i]);
                        $fdisplay(log_file, "    Actual FP16[%0d]:  0x%h", 2*i, actual_result_fp16[2*i]);
                        $fdisplay(log_file, "    Expected FP16[%0d]: 0x%h", 2*i, expected_result_fp16[2*i]);
                        $fdisplay(log_file, "    Actual FP16[%0d]:  0x%h", 2*i+1, actual_result_fp16[2*i+1]);
                        $fdisplay(log_file, "    Expected FP16[%0d]: 0x%h", 2*i+1, expected_result_fp16[2*i+1]);
                        error_count = error_count + 1;
                        passed = 1'b0;
                    end
                end
            end
        endcase
        
        if (passed) begin
            $fdisplay(log_file, "\n  RESULT: PASSED!");
            $display("  [PASS] Test %0d: %0s", test_num, test_name);
        end else begin
            $fdisplay(log_file, "\n  RESULT: FAILED! (%0d out of %0d lanes failed)", error_count, 
                     (conversion_mode == 2'd0 || conversion_mode == 2'd3) ? 8 : 4);
            $display("  [FAIL] Test %0d: %0s (%0d lanes failed)", test_num, test_name, error_count);
        end
        $fdisplay(log_file, "========================================\n");
    end
    endtask

    // 运行并检查任务（FP16toFP16）
    task run_and_check_fp16tofp16;
        input [1023:0] test_name;
        input [15:0]   new_inputs [0:7];
        reg            passed;
        integer        i;
    begin
        test_count = test_count + 1;
        conversion_mode = 2'd0;
        
        // 更新测试数据
        for (i = 0; i < 8; i = i + 1) begin
            test_inputs_fp16[i] = new_inputs[i];
        end
        
        // 计算期望值
        compute_expected_values_fp16tofp16();
        
        // 加载数据并触发计算
        calculate_and_load_regs_fp16tofp16(new_inputs);
        @(posedge clk); // 第1个周期：输入进入流水线 (Stage 1)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+1: cru_r=0x%h, instruction_valid=%b, mode_fp16_to_fp16=%b", 
                 debug_cru_r, debug_instruction_valid, debug_mode_fp16_to_fp16);
        @(posedge clk); // 第2个周期：计算完成 (Stage 2)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+2: output_data_r=0x%h, cru_r2=0x%h", 
                 debug_output_data_r, dut.cru_r2);
        @(posedge clk); // 第3个周期：输出稳定 (Stage 3)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+3: dr_d_r=0x%h", dut.dr_d_r);
        @(posedge clk); // 第4个周期：确保输出稳定
        
        // 提取实际结果
        extract_actual_results_fp16tofp16();
        
        // 检查结果
        check_results(test_name, test_count, passed);
        
        if (passed) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
        end
    end
    endtask

    // 运行并检查任务（FP32toFP32）
    task run_and_check_fp32tofp32;
        input [1023:0] test_name;
        input [31:0]   new_inputs [0:3];
        reg            passed;
        integer        i;
    begin
        test_count = test_count + 1;
        conversion_mode = 2'd1;
        
        // 更新测试数据
        for (i = 0; i < 4; i = i + 1) begin
            test_inputs_fp32[i] = new_inputs[i];
        end
        
        // 计算期望值
        compute_expected_values_fp32tofp32();
        
        // 加载数据并触发计算
        calculate_and_load_regs_fp32tofp32(new_inputs);
        @(posedge clk); // 第1个周期：输入进入流水线 (Stage 1)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+1: cru_r=0x%h, instruction_valid=%b, mode_fp32_to_fp32=%b", 
                 debug_cru_r, debug_instruction_valid, debug_mode_fp32_to_fp32);
        @(posedge clk); // 第2个周期：计算完成 (Stage 2)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+2: output_data_r=0x%h, cru_r2=0x%h", 
                 debug_output_data_r, dut.cru_r2);
        @(posedge clk); // 第3个周期：输出稳定 (Stage 3)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+3: dr_d_r=0x%h", dut.dr_d_r);
        @(posedge clk); // 第4个周期：确保输出稳定
        
        // 提取实际结果
        extract_actual_results_fp32tofp32();
        
        // 检查结果
        check_results(test_name, test_count, passed);
        
        if (passed) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
        end
    end
    endtask

    // 运行并检查任务（FP16toFP32）
    task run_and_check_fp16tofp32;
        input [1023:0] test_name;
        input [15:0]   new_inputs [0:7];
        input          src_pos_in;
        reg            passed;
        integer        i;
    begin
        test_count = test_count + 1;
        conversion_mode = 2'd2;
        
        // 更新测试数据
        for (i = 0; i < 8; i = i + 1) begin
            test_inputs_fp16[i] = new_inputs[i];
        end
        
        // 加载数据并触发计算
        calculate_and_load_regs_fp16tofp32(new_inputs, src_pos_in);
        @(posedge clk); // 第1个周期：输入进入流水线 (Stage 1)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+1: cru_r=0x%h, instruction_valid=%b, mode_fp16_to_fp32=%b, src_pos=%b", 
                 debug_cru_r, debug_instruction_valid, debug_mode_fp16_to_fp32, debug_src_precision);
        @(posedge clk); // 第2个周期：计算完成 (Stage 2)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+2: output_data_r=0x%h, cru_r2=0x%h", 
                 debug_output_data_r, dut.cru_r2);
        @(posedge clk); // 第3个周期：输出稳定 (Stage 3)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+3: dr_d_r=0x%h", dut.dr_d_r);
        @(posedge clk); // 第4个周期：确保输出稳定
        
        // 计算期望值（需要在设置cru_fptofp之后）
        compute_expected_values_fp16tofp32();
        
        // 提取实际结果
        extract_actual_results_fp16tofp32();
        
        // 检查结果
        check_results(test_name, test_count, passed);
        
        if (passed) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
        end
    end
    endtask

    // 运行并检查任务（FP32toFP16）
    task run_and_check_fp32tofp16;
        input [1023:0] test_name;
        input [31:0]   new_inputs [0:3];
        input          dest_pos_in;
        reg            passed;
        integer        i;
    begin
        test_count = test_count + 1;
        conversion_mode = 2'd3;
        
        // 更新测试数据
        for (i = 0; i < 4; i = i + 1) begin
            test_inputs_fp32[i] = new_inputs[i];
        end
        
        // 计算期望值
        compute_expected_values_fp32tofp16();
        
        // 加载数据并触发计算
        calculate_and_load_regs_fp32tofp16(new_inputs, dest_pos_in);
        @(posedge clk); // 第1个周期：输入进入流水线 (Stage 1)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+1: cru_r=0x%h, instruction_valid=%b, mode_fp32_to_fp16=%b", 
                 debug_cru_r, debug_instruction_valid, debug_mode_fp32_to_fp16);
        @(posedge clk); // 第2个周期：计算完成 (Stage 2)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+2: output_data_r=0x%h, cru_r2=0x%h", 
                 debug_output_data_r, dut.cru_r2);
        @(posedge clk); // 第3个周期：输出稳定 (Stage 3)
        // 监控中间变量
        $fdisplay(log_file, "  [DEBUG] T+3: dr_d_r=0x%h", dut.dr_d_r);
        @(posedge clk); // 第4个周期：确保输出稳定
        
        // 提取实际结果
        extract_actual_results_fp32tofp16();
        
        // 检查结果
        check_results(test_name, test_count, passed);
        
        if (passed) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
        end
    end
    endtask

    // 固定测试任务
    task fixed_test_suite;
        begin
            reg [1023:0] test_names [0:19];
            reg [15:0]  test_inputs_fp16_array [0:19][0:7];
            reg [31:0]  test_inputs_fp32_array [0:19][0:3];
            reg [1:0]   test_modes [0:19];
            reg         test_src_pos [0:19];
            reg         test_dest_pos [0:19];
            integer i, j;
            
            $fdisplay(log_file, "\n\n========================================");
            $fdisplay(log_file, "  FIXED TEST SUITE - Starting 20 test cases");
            $fdisplay(log_file, "========================================\n");
            $display("\n[Fixed Test Suite] Starting 20 test cases...");
            
            // 测试用例 0: FP16toFP16 - 全零
            test_names[0] = "FP16toFP16: All Zeros";
            test_modes[0] = 2'd0;
            for (i = 0; i < 8; i = i + 1) begin
                test_inputs_fp16_array[0][i] = 16'h0000;
            end
            
            // 测试用例 1: FP16toFP16 - 1.0
            test_names[1] = "FP16toFP16: All 1.0";
            test_modes[1] = 2'd0;
            for (i = 0; i < 8; i = i + 1) begin
                test_inputs_fp16_array[1][i] = 16'h3C00; // 1.0 in FP16
            end
            
            // 测试用例 2: FP32toFP32 - 全零
            test_names[2] = "FP32toFP32: All Zeros";
            test_modes[2] = 2'd1;
            for (i = 0; i < 4; i = i + 1) begin
                test_inputs_fp32_array[2][i] = 32'h00000000;
            end
            
            // 测试用例 3: FP32toFP32 - 1.0
            test_names[3] = "FP32toFP32: All 1.0";
            test_modes[3] = 2'd1;
            for (i = 0; i < 4; i = i + 1) begin
                test_inputs_fp32_array[3][i] = 32'h3F800000; // 1.0 in FP32
            end
            
            // 测试用例 4: FP16toFP32 - 1.0
            test_names[4] = "FP16toFP32: All 1.0";
            test_modes[4] = 2'd2;
            test_src_pos[4] = 1'b0;
            for (i = 0; i < 8; i = i + 1) begin
                test_inputs_fp16_array[4][i] = 16'h3C00; // 1.0 in FP16
            end
            
            // 测试用例 5: FP32toFP16 - 1.0
            test_names[5] = "FP32toFP16: All 1.0";
            test_modes[5] = 2'd3;
            test_dest_pos[5] = 1'b0;
            for (i = 0; i < 4; i = i + 1) begin
                test_inputs_fp32_array[5][i] = 32'h3F800000; // 1.0 in FP32
            end
            
            // 测试用例 6-19: 更多测试用例...
            for (j = 6; j < 20; j = j + 1) begin
                test_names[j] = $sformatf("Test Case %0d", j);
                test_modes[j] = j % 4;
                test_src_pos[j] = 1'b0;
                test_dest_pos[j] = 1'b0;
                for (i = 0; i < 8; i = i + 1) begin
                    test_inputs_fp16_array[j][i] = 16'h3C00 + (i * j);
                end
                for (i = 0; i < 4; i = i + 1) begin
                    test_inputs_fp32_array[j][i] = 32'h3F800000 + (i * j);
                end
            end
            
            // 执行所有固定测试用例
            for (j = 0; j < 20; j = j + 1) begin
                case (test_modes[j])
                    2'd0: run_and_check_fp16tofp16(test_names[j], test_inputs_fp16_array[j]);
                    2'd1: run_and_check_fp32tofp32(test_names[j], test_inputs_fp32_array[j]);
                    2'd2: run_and_check_fp16tofp32(test_names[j], test_inputs_fp16_array[j], test_src_pos[j]);
                    2'd3: run_and_check_fp32tofp16(test_names[j], test_inputs_fp32_array[j], test_dest_pos[j]);
                endcase
                reset_dut();
                @(posedge clk);
            end
            
            $fdisplay(log_file, "\n\n========================================");
            $fdisplay(log_file, "  FIXED TEST SUITE - Completed");
            $fdisplay(log_file, "========================================\n");
            $display("[Fixed Test Suite] Completed 20 test cases");
        end
    endtask

    // 随机测试任务
    task random_test_suite;
        integer random_seed_gen;
        integer rand_pass_count;
        integer rand_fail_count;
        integer i, j;
        integer prev_pass_count;
        integer prev_test_count;
        reg [15:0] rand_fp16 [0:7];
        reg [31:0] rand_fp32 [0:3];
        reg [1:0]  rand_mode;
        reg        rand_src_pos, rand_dest_pos;
    begin
        random_seed_gen = RANDOM_SEED;
        rand_pass_count = 0;
        rand_fail_count = 0;
        
        $fdisplay(log_file, "\n\n========================================");
        $fdisplay(log_file, "  RANDOM TEST SUITE - Starting %0d test cases", NUM_RANDOM_TESTS);
        $fdisplay(log_file, "========================================\n");
        $display("\n[Random Test Suite] Starting %0d test cases...", NUM_RANDOM_TESTS);
        
        for (i = 0; i < NUM_RANDOM_TESTS; i = i + 1) begin
            // 随机选择转换模式
            rand_mode = $random(random_seed_gen) % 4;
            rand_src_pos = $random(random_seed_gen) % 2;
            rand_dest_pos = $random(random_seed_gen) % 2;
            
            // 生成随机输入
            for (j = 0; j < 8; j = j + 1) begin
                rand_fp16[j] = $random(random_seed_gen);
            end
            for (j = 0; j < 4; j = j + 1) begin
                rand_fp32[j] = $random(random_seed_gen);
            end
            
            // 记录测试前的计数
            prev_pass_count = pass_count;
            prev_test_count = test_count;
            
            // 根据模式运行测试
            case (rand_mode)
                2'd0: run_and_check_fp16tofp16($sformatf("Random Test %0d", i+1), rand_fp16);
                2'd1: run_and_check_fp32tofp32($sformatf("Random Test %0d", i+1), rand_fp32);
                2'd2: run_and_check_fp16tofp32($sformatf("Random Test %0d", i+1), rand_fp16, rand_src_pos);
                2'd3: run_and_check_fp32tofp16($sformatf("Random Test %0d", i+1), rand_fp32, rand_dest_pos);
            endcase
            
            // 检查测试是否通过
            if (pass_count > prev_pass_count) begin
                rand_pass_count = rand_pass_count + 1;
            end else begin
                rand_fail_count = rand_fail_count + 1;
            end
            
            reset_dut();
            @(posedge clk);
            
            if ((i + 1) % 10000 == 0) begin
                $display("[Random Test] Progress: %0d/%0d (Pass: %0d, Fail: %0d)",
                         i + 1, NUM_RANDOM_TESTS, rand_pass_count, rand_fail_count);
            end
        end
        
        $fdisplay(log_file, "\n--- Random Test Suite Completed ---");
        $fdisplay(log_file, "Passed: %0d, Failed: %0d", rand_pass_count, rand_fail_count);
        $display("[Random Test Suite] Completed: Passed: %0d, Failed: %0d", rand_pass_count, rand_fail_count);
    end
    endtask

    // 主测试流程
    initial begin
        // 打开日志文件
        log_file = $fopen("fptofp_test.log", "w");
        if (!log_file) begin
            $display("ERROR: Cannot open log file!");
            $finish;
        end
        
        $fdisplay(log_file, "========================================");
        $fdisplay(log_file, "FPtoFP Array DPI Test Suite");
        $fdisplay(log_file, "Start Time: %0t", $time);
        $fdisplay(log_file, "========================================");
        
        // 初始化
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        smc_id = 5'd0;
        
        // 复位
        reset_dut();
        #(CLK_PERIOD * 2);
        
        // 运行固定测试
        fixed_test_suite();
        
        // 运行随机测试
        random_test_suite();
        
        // 最终摘要
        $fdisplay(log_file, "\n\n========================================");
        $fdisplay(log_file, "  FINAL TEST SUMMARY");
        $fdisplay(log_file, "========================================");
        $fdisplay(log_file, "Total Tests: %0d", test_count);
        $fdisplay(log_file, "Passed     : %0d", pass_count);
        $fdisplay(log_file, "Failed     : %0d", fail_count);
        $fdisplay(log_file, "Pass Rate  : %.2f%%", (test_count > 0) ? (100.0 * pass_count / test_count) : 0.0);
        $fdisplay(log_file, "End Time   : %0t", $time);
        $fdisplay(log_file, "========================================");
        
        $display("\n\n========================================");
        $display("  FINAL TEST SUMMARY");
        $display("========================================");
        $display("Total Tests: %0d", test_count);
        $display("Passed     : %0d", pass_count);
        $display("Failed     : %0d", fail_count);
        $display("Pass Rate  : %.2f%%", (test_count > 0) ? (100.0 * pass_count / test_count) : 0.0);
        $display("========================================");
        
        $fclose(log_file);
        #(CLK_PERIOD * 10);
        $finish;
    end

endmodule

