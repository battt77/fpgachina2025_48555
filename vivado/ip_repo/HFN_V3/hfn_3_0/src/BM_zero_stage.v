module BM_zero_stage #(
    parameter bitwidth = 16,
    parameter N = 8,
    parameter hidden_num = 16
) (
    input wire clk,
    input wire rstn,

    input wire [N*bitwidth-1:0] gradz,
    input wire [N*bitwidth-1:0] y1,
    input wire in_valid,
    input wire in_last,

    output wire [bitwidth-1:0] gradg_out,
    output wire gradg_out_valid,
    output wire gradg_out_last,

    output wire [bitwidth-1:0] gradb_out,
    output wire gradb_out_valid,
    output wire gradb_out_last
);

wire [bitwidth-1:0] gradz_sum;
wire gradz_sum_valid;
wire gradz_sum_last;
reduction_tree #(
    .bitwidth(bitwidth),
    .N(N)
) u_gradz_sum (
    .clk(clk),
    .rstn(rstn),
    .in(gradz),
    .in_valid(in_valid),
    .in_last(in_last),
    .out(gradz_sum),
    .out_valid(gradz_sum_valid),
    .out_last(gradz_sum_last)
);

wire [bitwidth-1:0] gradz_y1_sum;
wire gradz_y1_sum_valid;
wire gradz_y1_sum_last;
vector_multiply_reduction_tree #(
    .bitwidth(bitwidth),
    .N(N)
) u_gradz_y1_mult_sum (
    .clk(clk),
    .rstn(rstn),
    .in0(gradz),
    .in1(y1),
    .in_valid(in_valid),
    .in_last(in_last),
    .out(gradz_y1_sum),
    .out_valid(gradz_y1_sum_valid),
    .out_last(gradz_y1_sum_last)
);

wire [bitwidth-1:0] gradz_acc_out;
wire gradz_acc_valid;
wire gradz_acc_last;
floating_acc u_gradz_acc (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(gradz_sum_valid),
    .s_axis_a_tlast(gradz_sum_last),
    .s_axis_a_tdata(gradz_sum),
    .m_axis_result_tvalid(gradz_acc_valid),
    .m_axis_result_tlast(gradz_acc_last),
    .m_axis_result_tdata(gradz_acc_out)
);

wire [bitwidth-1:0] gradz_y1_acc_out;
wire gradz_y1_acc_valid;
wire gradz_y1_acc_last;
floating_acc u_gradz_y1_acc (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(gradz_y1_sum_valid),
    .s_axis_a_tlast(gradz_y1_sum_last),
    .s_axis_a_tdata(gradz_y1_sum),
    .m_axis_result_tvalid(gradz_y1_acc_valid),
    .m_axis_result_tlast(gradz_y1_acc_last),
    .m_axis_result_tdata(gradz_y1_acc_out)
);

assign gradb_out = gradz_acc_out;
assign gradb_out_valid = gradz_acc_valid;
assign gradb_out_last = gradz_acc_last;

assign gradg_out = gradz_y1_acc_out;
assign gradg_out_valid = gradz_y1_acc_valid;
assign gradg_out_last = gradz_y1_acc_last;

endmodule