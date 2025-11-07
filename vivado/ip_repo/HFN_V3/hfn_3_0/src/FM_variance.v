module FM_variance #(
    parameter   bitwidth = 16,
    parameter   N = 8
) (
    input wire clk,
    input wire rstn,
    input wire mode,

    input wire [N*bitwidth-1:0] x,
    input wire x_valid,
    input wire x_last,

    input wire [bitwidth-1:0] C_param,
    input wire [bitwidth-1:0] H_param,
    input wire [bitwidth-1:0] E_param,

    output wire [N-1:0] max_index,
    output wire [N-1:0] min_index,
    output wire index_valid,
    output wire index_last,

    output wire [bitwidth-1:0] out,
    output wire out_valid,
    output wire out_last
);
    
assign index_valid = compare_tree_valid;
assign index_last = compare_tree_last;


wire [bitwidth-1:0] max_out;
wire [bitwidth-1:0] min_out;
wire compare_tree_valid;
wire compare_tree_last;
vector_compare_tree #(
    .bitwidth(bitwidth),
    .N(N)
) u_vector_compare_tree (
    .clk(clk),
    .rstn(rstn),
    .in(x),
    .in_valid(x_valid),
    .in_last(x_last),
    .max_out(max_out),
    .max_index(max_index),
    .min_out(min_out),
    .min_index(min_index),
    .out_valid(compare_tree_valid),
    .out_last(compare_tree_last)
);

wire [bitwidth-1:0] rms_out;
wire rms_valid;
wire rms_last;
vector_multiply_reduction_tree #(
    .bitwidth(bitwidth),
    .N(N)
) u_vector_multiply_reduction_tree (
    .clk(clk),
    .rstn(rstn),
    .in0(x),
    .in1(x),
    .in_valid(x_valid),
    .in_last(x_last),
    .out(rms_out),
    .out_valid(rms_valid),
    .out_last(rms_last)
);

wire [bitwidth-1:0] sub_out;
wire sub_valid;
wire sub_last;
floating_sub u_floating_sub (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(compare_tree_valid),
    .s_axis_a_tlast(compare_tree_last),
    .s_axis_a_tdata(max_out),
    .s_axis_b_tvalid(compare_tree_valid),
    .s_axis_b_tlast(compare_tree_last),
    .s_axis_b_tdata(min_out),
    .m_axis_result_tvalid(sub_valid),
    .m_axis_result_tlast(sub_last),
    .m_axis_result_tdata(sub_out)
);

wire [bitwidth-1:0] mux0_out;
wire mux0_valid;
wire mux0_last;
assign mux0_out = mode ? rms_out : sub_out;
assign mux0_valid = mode ? rms_valid : sub_valid;
assign mux0_last = mode ? rms_last : sub_last;


wire [bitwidth-1:0] acc_out;
wire acc_valid;
wire acc_last;
floating_acc u_floating_acc (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(mux0_valid),
    .s_axis_a_tlast(mux0_last),
    .s_axis_a_tdata(mux0_out),
    .m_axis_result_tvalid(acc_valid),
    .m_axis_result_tlast(acc_last),
    .m_axis_result_tdata(acc_out)
);

wire [bitwidth-1:0] mux1_out;
assign mux1_out = mode ? H_param : C_param;

wire [bitwidth-1:0] mult_out;
wire mult_valid;
wire mult_last;
floating_mult u_floating_mult (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(acc_valid),
    .s_axis_a_tlast(acc_last),
    .s_axis_a_tdata(acc_out),
    .s_axis_b_tvalid(acc_valid),
    .s_axis_b_tlast(acc_last),
    .s_axis_b_tdata(mux1_out),
    .m_axis_result_tvalid(mult_valid),
    .m_axis_result_tlast(mult_last),
    .m_axis_result_tdata(mult_out)
);

floating_add u_floating_add (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(mult_valid),
    .s_axis_a_tlast(mult_last),
    .s_axis_a_tdata(mult_out),
    .s_axis_b_tvalid(mult_valid),
    .s_axis_b_tlast(mult_last),
    .s_axis_b_tdata(E_param),
    .m_axis_result_tvalid(out_valid),
    .m_axis_result_tlast(out_last),
    .m_axis_result_tdata(out)
);



endmodule