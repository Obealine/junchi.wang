`timescale 1ns / 1ps
//=========================================================
//File     : user_register.v
//Moudule  : User Register Module
//Author   : Oliver
//Email    :
//Created  : 2025-11-20
//Modified : 2025-11-20
//Version  : v1.0
//=========================================================
//Description:
//  用户寄存器模块
//
//Features:
//  1. 支持多读多写端口（不同时对同一个bank做操作）
//  2. 支持写入和读取流水线
//
//Parameters:
//  SMC_ID                     : 片上模块ID
//  PARAM_UR_BYTE_CNT          : 用户寄存器字节数
//  PARAM_UR_WR_PORT_CNT       : 写端口数量
//  PARAM_UR_RD_PORT_CNT       : 读端口数量
//  PARAM_UR_BANK_BIT_NUM      : 寄存器BANK编码位数
//  PARAM_UR_BANK_ADDR_BIT_NUM : 寄存器BANK地址位数
//  PARAM_UR_WRI_PIPE_CNT      : 写入流水线深度
//  PARAM_UR_RDI_PIPE_CNT      : 读取指令流水线深度
//  PARAM_UR_RDO_PIPE_CNT      : 读取数据流水线深度
//
//IO Ports:
//  i_clk              : 时钟信号
//  i_rst_n            : 复位信号，低有效
//  i_smc_id           : 片上模块ID
//  o_ur_rdata_X       : 读端口X数据输出
//  i_uvr_wdata_X      : 写端口X数据输入
//  i_cru_ur           : 控制寄存器单元输入信号
//
//Dependencies:
//  None
//
//Called by:
//  Top Module
//=========================================================
//Revision History:
//Date        Version    Author      Description
//2025-11-20  v1.0       Oliver      Created file
//=========================================================
module user_register #(
    parameter SMC_ID                     = 5'd0,
    parameter PARAM_UR_BYTE_CNT          = 5'd16,
    parameter PARAM_UR_WR_PORT_CNT       = 3'd4,
    parameter PARAM_UR_RD_PORT_CNT       = 3'd4,
    parameter PARAM_UR_BANK_BIT_NUM      = 2'd3,
    parameter PARAM_UR_BANK_ADDR_BIT_NUM = 4'd8,
    parameter PARAM_UR_WRI_PIPE_CNT      = 1'd1,
    parameter PARAM_UR_RDI_PIPE_CNT      = 1'd1,
    parameter PARAM_UR_RDO_PIPE_CNT      = 1'd1
)(
    input  wire                       i_clk,
    input  wire                       i_rst_n,
    input  wire [4:0]                 i_smc_id,
    input  wire [160:0]               i_cru_ur,      //UR上行指令寄存器 1+(8+3)*4*2+16*4=161

    input  wire [127:0]               i_uvr_wdata_0,
    input  wire [127:0]               i_uvr_wdata_1,
    input  wire [127:0]               i_uvr_wdata_2,
    input  wire [127:0]               i_uvr_wdata_3,

    output wire [127:0]               o_ur_rdata_0,
    output wire [127:0]               o_ur_rdata_1,
    output wire [127:0]               o_ur_rdata_2,
    output wire [127:0]               o_ur_rdata_3,

    output reg  [160:0]               o_cru_ur      //留个空
);
// ==========================================================================
// 1. 控制信号提取
// ==========================================================================

// 控制寄存器单元有效信号
wire w_cru_ur_vld = (i_smc_id == SMC_ID) && i_cru_ur[0];

// 写入BANK编码
wire [2:0] w_ur_wbank_0 = i_cru_ur[3:1];
wire [2:0] w_ur_wbank_1 = i_cru_ur[6:4];
wire [2:0] w_ur_wbank_2 = i_cru_ur[9:7];
wire [2:0] w_ur_wbank_3 = i_cru_ur[12:10];

// 写入地址
wire [7:0] w_ur_waddr_0 = i_cru_ur[20:13];
wire [7:0] w_ur_waddr_1 = i_cru_ur[28:21];
wire [7:0] w_ur_waddr_2 = i_cru_ur[36:29];
wire [7:0] w_ur_waddr_3 = i_cru_ur[44:37];

// 读取BANK编码
wire [2:0] w_ur_rbank_0 = i_cru_ur[47:45];
wire [2:0] w_ur_rbank_1 = i_cru_ur[50:48];
wire [2:0] w_ur_rbank_2 = i_cru_ur[53:51];
wire [2:0] w_ur_rbank_3 = i_cru_ur[56:54];

// 读取地址
wire [7:0] w_ur_raddr_0 = i_cru_ur[64:57];
wire [7:0] w_ur_raddr_1 = i_cru_ur[72:65];
wire [7:0] w_ur_raddr_2 = i_cru_ur[80:73];
wire [7:0] w_ur_raddr_3 = i_cru_ur[88:81];

// 读取使能
wire w_ur_re_0 = i_cru_ur[89];
wire w_ur_re_1 = i_cru_ur[90];
wire w_ur_re_2 = i_cru_ur[91];
wire w_ur_re_3 = i_cru_ur[92];

// 写入使能
wire w_ur_we_0 = i_cru_ur[93];
wire w_ur_we_1 = i_cru_ur[94];
wire w_ur_we_2 = i_cru_ur[95];
wire w_ur_we_3 = i_cru_ur[96];

// 写入字节使能
wire [15:0] w_ur_wbe_0 = i_cru_ur[112:97];
wire [15:0] w_ur_wbe_1 = i_cru_ur[128:113];
wire [15:0] w_ur_wbe_2 = i_cru_ur[144:129];
wire [15:0] w_ur_wbe_3 = i_cru_ur[160:145];

// ==========================================================================
// 2. pipeline寄存
// ==========================================================================
// 写入流水线寄存
reg         r_we_pipe[0:7];
reg [10:0]  r_waddr_pipe[0:7];
reg [15:0]  r_wbe_pipe[0:7];
reg [127:0] r_wdata_pipe[0:7];

// 读取流水线寄存
reg         r_re_pipe[0:7];
reg [10:0]  r_raddr_pipe[0:7];
reg [127:0] r_rdata_pipe[0:7];

// 声明为wire类型，用于连续赋值
wire [7:0]   w_bank_ena;
wire [7:0]   w_bank_enb;
wire [15:0]  w_bank_wbea [0:7];
wire [7:0]   w_bank_addra [0:7];
wire [7:0]   w_bank_addrb [0:7];
wire [127:0] w_bank_dina [0:7];
wire [127:0] w_bank_doutb [0:7];

// 写入流水线寄存逻辑
integer i;
always @(posedge i_clk or negedge i_rst_n) begin
    if (!(i_rst_n && w_cru_ur_vld)) begin
        for (i = 0; i < 8; i = i + 1) begin
            r_we_pipe[i]     <= 1'b0;
            r_waddr_pipe[i]  <= 11'd0;
            r_wbe_pipe[i]    <= 16'd0;
            r_wdata_pipe[i]  <= 128'd0;
        end
    end else begin
        for (i = 0; i < 8; i = i + 1) begin
            if (w_ur_wbank_0 == i) begin
                r_we_pipe[i]     <= w_ur_we_0;
                r_waddr_pipe[i]  <= w_ur_waddr_0;
                r_wbe_pipe[i]    <= w_ur_wbe_0;
                r_wdata_pipe[i]  <= i_uvr_wdata_0;
            end else if (w_ur_wbank_1 == i) begin
                r_we_pipe[i]     <= w_ur_we_1;
                r_waddr_pipe[i]  <= w_ur_waddr_1;
                r_wbe_pipe[i]    <= w_ur_wbe_1;
                r_wdata_pipe[i]  <= i_uvr_wdata_1;
            end else if (w_ur_wbank_2 == i) begin
                r_we_pipe[i]     <= w_ur_we_2;
                r_waddr_pipe[i]  <= w_ur_waddr_2;
                r_wbe_pipe[i]    <= w_ur_wbe_2;
                r_wdata_pipe[i]  <= i_uvr_wdata_2;
            end else if (w_ur_wbank_3 == i) begin
                r_we_pipe[i]     <= w_ur_we_3;
                r_waddr_pipe[i]  <= w_ur_waddr_3;
                r_wbe_pipe[i]    <= w_ur_wbe_3;
                r_wdata_pipe[i]  <= i_uvr_wdata_3;
            end else begin
                r_we_pipe[i]     <= r_we_pipe[i];
                r_waddr_pipe[i]  <= r_waddr_pipe[i];
                r_wbe_pipe[i]    <= r_wbe_pipe[i];
                r_wdata_pipe[i]  <= r_wdata_pipe[i];
            end
        end
    end
end
// 读取流水线寄存逻辑
always @(posedge i_clk or negedge i_rst_n) begin
    if (!i_rst_n) begin
        for (i = 0; i < 8; i = i + 1) begin
            r_re_pipe[i]    <= 1'b0;
            r_raddr_pipe[i] <= 11'd0;
            r_rdata_pipe[i] <= 128'd0;
        end
    end else begin
        for (i = 0; i < 8; i = i + 1) begin
            if (w_ur_rbank_0 == i) begin
                r_re_pipe[i]    <= w_ur_re_0;
                r_raddr_pipe[i] <= w_ur_raddr_0;
                r_rdata_pipe[i] <= w_bank_doutb[i];
            end else if (w_ur_rbank_1 == i) begin
                r_re_pipe[i]    <= w_ur_re_1;
                r_raddr_pipe[i] <= w_ur_raddr_1;
                r_rdata_pipe[i] <= w_bank_doutb[i];
            end else if (w_ur_rbank_2 == i) begin
                r_re_pipe[i]    <= w_ur_re_2;
                r_raddr_pipe[i] <= w_ur_raddr_2;
                r_rdata_pipe[i] <= w_bank_doutb[i];
            end else if (w_ur_rbank_3 == i) begin
                r_re_pipe[i]    <= w_ur_re_3;
                r_raddr_pipe[i] <= w_ur_raddr_3;
                r_rdata_pipe[i] <= w_bank_doutb[i];
            end else begin
                r_re_pipe[i]    <= r_re_pipe[i];
                r_raddr_pipe[i] <= r_raddr_pipe[i];
                r_rdata_pipe[i] <= r_rdata_pipe[i];
            end
        end
    end
end

// ==========================================================================
// 3. 端口信号连接转换 - 使用连续赋值代替过程赋值
// ==========================================================================
genvar j;
generate
    for (j = 0; j < 8; j = j + 1) begin : bank_signals
        // 写入控制信号 - 使用连续赋值
        assign w_bank_ena[j] = w_cru_ur_vld ? (
            (w_ur_wbank_0 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_we_pipe[j] : w_ur_we_0) :
            (w_ur_wbank_1 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_we_pipe[j] : w_ur_we_1) :
            (w_ur_wbank_2 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_we_pipe[j] : w_ur_we_2) :
            (w_ur_wbank_3 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_we_pipe[j] : w_ur_we_3) : 1'b0
        ) : 1'b0;
        assign w_bank_wbea[j] = w_cru_ur_vld ? (
            (w_ur_wbank_0 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wbe_pipe[j] : w_ur_wbe_0) :
            (w_ur_wbank_1 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wbe_pipe[j] : w_ur_wbe_1) :
            (w_ur_wbank_2 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wbe_pipe[j] : w_ur_wbe_2) :
            (w_ur_wbank_3 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wbe_pipe[j] : w_ur_wbe_3) : 16'd0
        ) : 16'd0;
        assign w_bank_addra[j] = w_cru_ur_vld ? (
            (w_ur_wbank_0 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_waddr_pipe[j][7:0] : w_ur_waddr_0) :
            (w_ur_wbank_1 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_waddr_pipe[j][7:0] : w_ur_waddr_1) :
            (w_ur_wbank_2 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_waddr_pipe[j][7:0] : w_ur_waddr_2) :
            (w_ur_wbank_3 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_waddr_pipe[j][7:0] : w_ur_waddr_3) : 8'd0
        ) : 8'd0;
        assign w_bank_dina[j] = w_cru_ur_vld ? (
            (w_ur_wbank_0 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wdata_pipe[j] : i_uvr_wdata_0) :
            (w_ur_wbank_1 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wdata_pipe[j] : i_uvr_wdata_1) :
            (w_ur_wbank_2 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wdata_pipe[j] : i_uvr_wdata_2) :
            (w_ur_wbank_3 == j) ? (PARAM_UR_WRI_PIPE_CNT ? r_wdata_pipe[j] : i_uvr_wdata_3) : 128'd0
        ) : 128'd0;
        // 读取控制信号 - 使用连续赋值
        assign w_bank_enb[j] = w_cru_ur_vld ? (
            (w_ur_rbank_0 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_re_pipe[j] : w_ur_re_0) :
            (w_ur_rbank_1 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_re_pipe[j] : w_ur_re_1) :
            (w_ur_rbank_2 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_re_pipe[j] : w_ur_re_2) :
            (w_ur_rbank_3 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_re_pipe[j] : w_ur_re_3) : 1'b0
        ) : 1'b0;
        assign w_bank_addrb[j] = w_cru_ur_vld ? (
            (w_ur_rbank_0 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_raddr_pipe[j][7:0] : w_ur_raddr_0) :
            (w_ur_rbank_1 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_raddr_pipe[j][7:0] : w_ur_raddr_1) :
            (w_ur_rbank_2 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_raddr_pipe[j][7:0] : w_ur_raddr_2) :
            (w_ur_rbank_3 == j) ? (PARAM_UR_RDI_PIPE_CNT ? r_raddr_pipe[j][7:0] : w_ur_raddr_3) : 8'd0
        ) : 8'd0;
    end
endgenerate

// 输出选择
assign o_ur_rdata_0 = w_cru_ur_vld ? (PARAM_UR_RDO_PIPE_CNT ? r_rdata_pipe[w_ur_rbank_0] : w_bank_doutb[w_ur_rbank_0]) : 128'b0;
assign o_ur_rdata_1 = w_cru_ur_vld ? (PARAM_UR_RDO_PIPE_CNT ? r_rdata_pipe[w_ur_rbank_1] : w_bank_doutb[w_ur_rbank_1]) : 128'b0;
assign o_ur_rdata_2 = w_cru_ur_vld ? (PARAM_UR_RDO_PIPE_CNT ? r_rdata_pipe[w_ur_rbank_2] : w_bank_doutb[w_ur_rbank_2]) : 128'b0;
assign o_ur_rdata_3 = w_cru_ur_vld ? (PARAM_UR_RDO_PIPE_CNT ? r_rdata_pipe[w_ur_rbank_3] : w_bank_doutb[w_ur_rbank_3]) : 128'b0;

always @(posedge i_clk or negedge i_rst_n) begin
    if (!i_rst_n) begin
        o_cru_ur <= 161'd0;
    end else begin
        o_cru_ur <= i_cru_ur; //直接传递输入信号
    end
end

// ==========================================================================
// 4. 实例化BRAM BANK
// ==========================================================================
genvar x;
generate
    for (x = 0; x < 8; x = x + 1) begin : bram_bank_inst
        bank_sram u_bank_sram (
            .clk      (i_clk),
            .ena      (w_bank_ena[x]),
            .enb      (w_bank_enb[x]),
            .wea      (w_bank_wbea[x]),
            .addra    (w_bank_addra[x]),
            .addrb    (w_bank_addrb[x]),
            .dina     (w_bank_dina[x]),
            .doutb    (w_bank_doutb[x])
        );
    end
endgenerate

endmodule