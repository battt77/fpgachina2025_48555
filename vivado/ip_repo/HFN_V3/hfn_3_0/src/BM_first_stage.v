module BM_first_stage #(
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
    input wire mode, //0: hfln; 1: rms

    input wire [N*bitwidth-1:0] y2,
    input wire y2_in_valid,
    input wire y2_in_last,

    input wire [bitwidth-1:0] H_param,
    input wire [bitwidth-1:0] C_param,

    output wire gradw_out_last,
    output wire [N*bitwidth-1:0] gradw_y2_mult_out,
    output wire [bitwidth-1:0] gradb_add_gradg,
    output wire [bitwidth-1:0] gradb_sub_gradg,
    output wire [bitwidth-1:0] gradb,
    output wire out_valid,
    output wire out_last,
    output wire dz2start
);

wire [bitwidth-1:0] zero_stage_gradg_out;
wire zero_stage_gradg_valid;
wire zero_stage_gradg_last;

wire [bitwidth-1:0] zero_stage_gradb_out;
wire zero_stage_gradb_valid;
wire zero_stage_gradb_last;
BM_zero_stage #(
    .bitwidth(bitwidth),
    .N(N),
    .hidden_num(hidden_num)
) u_BM_zero_stage (
    .clk(clk),
    .rstn(rstn),
    .gradz(gradz),
    .y1(y1),
    .in_valid(in_valid),
    .in_last(in_last),
    .gradg_out(zero_stage_gradg_out),
    .gradg_out_valid(zero_stage_gradg_valid),
    .gradg_out_last(zero_stage_gradg_last),
    .gradb_out(zero_stage_gradb_out),
    .gradb_out_valid(zero_stage_gradb_valid),
    .gradb_out_last(zero_stage_gradb_last)
);

wire [bitwidth-1:0] gradb_out;
wire gradb_out_valid;
wire gradb_out_last;
floating_mult u_gradb_mult (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(zero_stage_gradb_valid),
    .s_axis_a_tlast(zero_stage_gradb_last),
    .s_axis_a_tdata(zero_stage_gradb_out),
    .s_axis_b_tvalid(zero_stage_gradb_valid),
    .s_axis_b_tlast(zero_stage_gradb_last),
    .s_axis_b_tdata(H_param),
    .m_axis_result_tvalid(gradb_out_valid),
    .m_axis_result_tlast(gradb_out_last),
    .m_axis_result_tdata(gradb_out)
);

wire [bitwidth-1:0] gradw_out;
wire gradw_out_valid;
//wire gradw_out_last;
wire [bitwidth-1:0] gradw_mult;
assign gradw_mult = (mode == 0) ? C_param : H_param;
floating_mult u_gradw_mult (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(zero_stage_gradg_valid),
    .s_axis_a_tlast(zero_stage_gradg_last),
    .s_axis_a_tdata(zero_stage_gradg_out),
    .s_axis_b_tvalid(zero_stage_gradg_valid),
    .s_axis_b_tlast(zero_stage_gradg_last),
    .s_axis_b_tdata(gradw_mult),
    .m_axis_result_tvalid(gradw_out_valid),
    .m_axis_result_tlast(gradw_out_last), //output -> y2_addr_cnt_en
    .m_axis_result_tdata(gradw_out)
);

reg [bitwidth-1:0] gradw_out_r;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        gradw_out_r <= 0;
    end
    else if (gradw_out_last) begin
        gradw_out_r <= gradw_out;
    end
end

///for rms
//wire [N*bitwidth-1:0] gradw_y2_mult_out;
wire gradw_y2_mult_valid;
wire gradw_y2_mult_last;
vector_mult_vs #(
    .bitwidth   (bitwidth),
    .N          (N)
) u_vector_mult_vs_gradw_y2(
    .clk(clk),
    .rstn(rstn),
    .in0(y2),
    .in1(gradw_out_r),
    .in_valid(y2_in_valid),
    .in_last(y2_in_last),
    .out(gradw_y2_mult_out),
    .out_valid(gradw_y2_mult_valid),
    .out_last(gradw_y2_mult_last)
);

///detect rising edge of gradw_y2_mult_valid
reg gradw_y2_mult_valid_r;
wire gradw_y2_mult_valid_posedge;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        gradw_y2_mult_valid_r <= 0;
    end
    else begin
        gradw_y2_mult_valid_r <= gradw_y2_mult_valid;
    end
end
assign gradw_y2_mult_valid_posedge = gradw_y2_mult_valid & (~gradw_y2_mult_valid_r);

reg [bitwidth-1:0] gradb_out_r;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        gradb_out_r <= 0;
    end
    else if (gradb_out_last) begin
        gradb_out_r <= gradb_out;
    end
end

wire [bitwidth-1:0] grad_add_result;
wire grad_add_result_valid;
wire grad_add_result_last;
floating_add u_grad_add (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(gradw_out_last),
    .s_axis_a_tlast(gradw_out_last),
    .s_axis_a_tdata(gradb_out_r),
    .s_axis_b_tvalid(gradw_out_last),
    .s_axis_b_tlast(gradw_out_last),
    .s_axis_b_tdata(gradw_out),
    .m_axis_result_tvalid(grad_add_result_valid),
    .m_axis_result_tlast(grad_add_result_last),
    .m_axis_result_tdata(grad_add_result)
);
        
wire [bitwidth-1:0] grad_sub_result; 
wire grad_sub_result_valid;
wire grad_sub_result_last;
floating_sub u_grad_sub (
    .aclk(clk),
    .aresetn(rstn),
    .s_axis_a_tvalid(gradw_out_last),
    .s_axis_a_tlast(gradw_out_last),
    .s_axis_a_tdata(gradb_out_r),
    .s_axis_b_tvalid(gradw_out_last),
    .s_axis_b_tlast(gradw_out_last),
    .s_axis_b_tdata(gradw_out),
    .m_axis_result_tvalid(grad_sub_result_valid),
    .m_axis_result_tlast(grad_sub_result_last),
    .m_axis_result_tdata(grad_sub_result)
);

wire [bitwidth-1:0] gradb_add_gradg_result;
wire [bitwidth-1:0] gradb_sub_gradg_result;
wire [bitwidth-1:0] gradb_result;

assign gradb_add_gradg_result = (grad_add_result_last & grad_sub_result_last) ? grad_add_result : 0;
assign gradb_sub_gradg_result = (grad_add_result_last & grad_sub_result_last) ? grad_sub_result : 0;
assign gradb_result = (grad_add_result_last & grad_sub_result_last) ? gradb_out_r : 0;

assign gradb_add_gradg = (mode == 0) ? gradb_add_gradg_result: 0;
assign gradb_sub_gradg = (mode == 0) ? gradb_sub_gradg_result : 0;
assign gradb =  (mode == 0) ?  gradb_result : 0;
assign out_valid = (mode == 0)? (grad_add_result_valid & grad_sub_result_valid) : gradw_y2_mult_valid;
assign out_last = (mode == 0)? (grad_add_result_last & grad_sub_result_last) : gradw_y2_mult_last;

assign dz2start = (out_last & (~mode)) | (gradw_y2_mult_valid_posedge & mode);

endmodule

