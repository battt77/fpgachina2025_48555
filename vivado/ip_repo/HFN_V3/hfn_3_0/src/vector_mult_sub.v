module vector_mult_sub #(
    parameter bitwidth = 16,
    parameter N = 8
) (
    input wire clk,
    input wire rstn,
    
    input wire [N*bitwidth-1:0]  in0,
    input wire [bitwidth-1:0]  in1,
    input wire [bitwidth-1:0]  in2,
    input wire in_valid,
    input wire in_last,

    output wire [N*bitwidth-1:0]   out,
    output wire out_valid,
    output wire out_last
);

    wire [N-1:0] stage_valid ;
    wire [N-1:0] stage_last;
    
    genvar i,j;
    generate
        for (i = 0; i < N; i = i + 1) begin: stage0
            floating_mult_sub u_mult_sub_0 (
                .aclk(clk),
                .aresetn(rstn),
                .s_axis_a_tvalid(in_valid),
                .s_axis_a_tlast(in_last),
                .s_axis_a_tdata(in0[i*bitwidth +: bitwidth]),
                .s_axis_b_tvalid(in_valid),
                .s_axis_b_tlast(in_last),
                .s_axis_b_tdata(in1),
                .s_axis_c_tvalid(in_valid),
                .s_axis_c_tlast(in_last),
                .s_axis_c_tdata(in2),
                .m_axis_result_tvalid(stage_valid[i]),
                .m_axis_result_tlast(stage_last[i]),
                .m_axis_result_tdata(out[i*bitwidth +: bitwidth])
            );
        end
    endgenerate
    
    assign out_valid = &stage_valid;
    assign out_last = &stage_last;

endmodule