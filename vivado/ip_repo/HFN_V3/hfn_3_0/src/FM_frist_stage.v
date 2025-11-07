module FM_first_stage #(
    parameter   bitwidth = 16,
    parameter   mean_N = 8,
    parameter   std_N = 8
) (
    input wire clk,
    input wire rstn,
    input wire mode,

    input wire [mean_N*bitwidth-1:0] x_mean,
    input wire x_mean_valid,
    input wire x_mean_last,

    input wire [std_N*bitwidth-1:0] x_std,
    input wire x_std_valid,
    input wire x_std_last,

    input wire [bitwidth-1:0] H_param,
    input wire [bitwidth-1:0] C_param,
    input wire [bitwidth-1:0] E_param,

    output wire [std_N-1:0] max_index,
    output wire [std_N-1:0] min_index,
    output wire index_valid,
    output wire index_last,

    output wire [bitwidth-1:0] mean_variance_out,
    output wire [bitwidth-1:0] one_variance_out,
    output wire out_valid,
    output wire out_last
);
    
wire [bitwidth-1:0] mean_out;
wire mean_out_valid;
wire mean_out_last;
FM_mean #(
    .bitwidth(bitwidth),
    .N(mean_N)
) u_FM_mean (
    .clk(clk),
    .rstn(rstn),
    .in(x_mean),
    .in_valid(x_mean_valid),
    .in_last(x_mean_last),
    .H_param(H_param),
    .out(mean_out),
    .out_valid(mean_out_valid),
    .out_last(mean_out_last)
);

reg [bitwidth-1:0] variance_out_r;
wire [bitwidth-1:0] variance_out;
wire variance_out_valid;
wire variance_out_last;
FM_variance #(
    .bitwidth(bitwidth),
    .N(std_N)
) u_FM_variance (
    .clk(clk),
    .rstn(rstn),
    .mode(mode),
    .x(x_std),
    .x_valid(x_std_valid),
    .x_last(x_std_last),
    .C_param(C_param),
    .H_param(H_param),
    .E_param(E_param),
    .max_index(max_index),
    .min_index(min_index),
    .index_valid(index_valid),
    .index_last(index_last),
    .out(variance_out),
    .out_valid(variance_out_valid),
    .out_last(variance_out_last)
);

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        variance_out_r <= 0;
    end 
    else if (variance_out_last) begin
        variance_out_r <= variance_out;
    end
end

wire [bitwidth-1:0] floating_rec_squ_root_out;
reg [bitwidth-1:0] floating_rec_squ_root_out_r;
wire floating_rec_squ_root_valid;
wire floating_rec_squ_root_last;
floating_rec_squ_root _floating_rec_squ_root (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(variance_out_valid),
    .s_axis_a_tlast(variance_out_last),
    .s_axis_a_tdata(variance_out),
    .m_axis_result_tvalid(floating_rec_squ_root_valid),
    .m_axis_result_tlast(floating_rec_squ_root_last),
    .m_axis_result_tdata(floating_rec_squ_root_out)
);

wire [bitwidth-1:0] floating_rec_out;
reg [bitwidth-1:0] floating_rec_out_r;
wire floating_rec_valid;
wire floating_rec_last;
floating_rec _floating_rec (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(variance_out_valid),
    .s_axis_a_tlast(variance_out_last),
    .s_axis_a_tdata(variance_out),
    .m_axis_result_tvalid(floating_rec_valid),
    .m_axis_result_tlast(floating_rec_last),
    .m_axis_result_tdata(floating_rec_out)
);

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        floating_rec_out_r <= 0;
    end 
    else if (floating_rec_last) begin
        floating_rec_out_r <= floating_rec_out;
    end
end

wire [bitwidth-1:0] mean_variance_out_0;
wire div_r_valid;
wire div_r_last;
// 上半部分
floating_div _floating_div (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(mean_out_valid),
    .s_axis_a_tlast(mean_out_last),
    .s_axis_a_tdata(mean_out),
    .s_axis_b_tvalid(mean_out_valid),
    .s_axis_b_tlast(mean_out_last),
    .s_axis_b_tdata(variance_out_r),
    .m_axis_result_tvalid(div_r_valid),
    .m_axis_result_tlast(div_r_last),
    .m_axis_result_tdata(mean_variance_out_0)
);
assign mean_variance_out = mode ? 0 : mean_variance_out_0;

// 下半部分
wire [bitwidth-1:0] one_variance_out_0;
wire [bitwidth-1:0] one_square_out;
assign one_variance_out_0 = out_last ? floating_rec_out_r : 0;
assign one_square_out = floating_rec_squ_root_last ? floating_rec_squ_root_out : 0;
assign one_variance_out = mode ? one_square_out :one_variance_out_0;
assign out_valid = mode ? floating_rec_squ_root_valid : div_r_valid;
assign out_last = mode ? floating_rec_squ_root_last : div_r_last;

endmodule