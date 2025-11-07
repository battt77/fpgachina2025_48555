module FM_mean #(
    parameter bitwidth = 16,
    parameter N = 8
) (
    input wire clk,
    input wire rstn,

    input wire [N*bitwidth-1:0] in,
    input wire in_valid,
    input wire in_last,

    input wire [bitwidth-1:0] H_param,

    output wire [bitwidth-1:0] out,
    output wire out_valid,
    output wire out_last
);

wire [bitwidth-1:0] reduction_tree_out;
wire reduction_tree_valid;
wire reduction_tree_last;

reduction_tree #(
    .bitwidth(bitwidth),
    .N(N)
) u_reduction_tree (
    .clk(clk),
    .rstn(rstn),
    .in(in),
    .in_valid(in_valid),
    .in_last(in_last),
    .out(reduction_tree_out),
    .out_valid(reduction_tree_valid),
    .out_last(reduction_tree_last)
);

wire [bitwidth-1:0] acc_out;
wire acc_valid;
wire acc_last;
floating_acc u_floating_acc (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(reduction_tree_valid),
    .s_axis_a_tlast(reduction_tree_last),
    .s_axis_a_tdata(reduction_tree_out),
    .m_axis_result_tvalid(acc_valid),
    .m_axis_result_tlast(acc_last),
    .m_axis_result_tdata(acc_out)
);

floating_mult u_floating_mult (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(acc_valid),
    .s_axis_a_tlast(acc_last),
    .s_axis_a_tdata(acc_out),
    .s_axis_b_tvalid(acc_valid),
    .s_axis_b_tlast(acc_last),
    .s_axis_b_tdata(H_param),
    .m_axis_result_tvalid(out_valid),
    .m_axis_result_tlast(out_last),
    .m_axis_result_tdata(out)
);



    
endmodule