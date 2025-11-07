module vector_multiply_reduction_tree #(
    parameter bitwidth = 16,
    parameter N = 8
) (
    input wire clk,
    input wire rstn,
    
    input wire [N*bitwidth-1:0]  in0,
    input wire [N*bitwidth-1:0]  in1,
    input wire in_valid,
    input wire in_last,

    output wire [bitwidth-1:0]   out,
    output wire out_valid,
    output wire out_last
);

    localparam ADD_STAGES = $clog2(N) - 1;  // Stage 0已经做了一次归约，还需要log2(N)-1级
    wire [bitwidth-1:0] mult_data [0:N-1];
    wire [bitwidth-1:0] stage_data [0:ADD_STAGES][0:N-1];

    wire [N-1:0] mult_valid;
    wire [N-1:0] mult_last;
    wire [N-1:0] stage_valid [0:ADD_STAGES];
    wire [N-1:0] stage_last  [0:ADD_STAGES];
    
    genvar i,j;
    generate
        for (i = 0; i < N; i = i + 1) begin: stage_mult
            floating_mult u_mult_0 (
                .aclk(clk),
                .aresetn(rstn),
                .s_axis_a_tvalid(in_valid),
                .s_axis_a_tlast(in_last),
                .s_axis_a_tdata(in0[i*bitwidth +: bitwidth]),
                .s_axis_b_tvalid(in_valid),
                .s_axis_b_tlast(in_last),
                .s_axis_b_tdata(in1[i*bitwidth +: bitwidth]),
                .m_axis_result_tvalid(mult_valid[i]),
                .m_axis_result_tlast(mult_last[i]),
                .m_axis_result_tdata(mult_data[i])
            );
        end
    endgenerate


    generate
        for (i = 0; i < N/2; i = i + 1) begin: stage0
            floating_add u_mult_0 (
                .aclk(clk),
                .aresetn(rstn),
                .s_axis_a_tvalid(mult_valid[2*i]),
                .s_axis_a_tlast(mult_last[2*i]),
                .s_axis_a_tdata(mult_data[2*i]),
                .s_axis_b_tvalid(mult_valid[2*i+1]),
                .s_axis_b_tlast(mult_last[2*i+1]),
                .s_axis_b_tdata(mult_data[2*i+1]),
                .m_axis_result_tvalid(stage_valid[0][i]),
                .m_axis_result_tlast(stage_last[0][i]),
                .m_axis_result_tdata(stage_data[0][i])
            );
        end
    endgenerate
    
    genvar stage;
    generate
        for (stage = 1; stage <= ADD_STAGES; stage = stage + 1) begin: add_tree_stage
            localparam NUM_NODES = N >> 1 >> stage; 
            for (j = 0; j < NUM_NODES; j = j + 1) begin: add_node
                floating_add u_add_node (
                    .aclk(clk),
                    .aresetn(rstn),
                    .s_axis_a_tvalid(stage_valid[stage-1][2*j]),
                    .s_axis_a_tlast(stage_last[stage-1][2*j]),
                    .s_axis_a_tdata(stage_data[stage-1][2*j]),
                    .s_axis_b_tvalid(stage_valid[stage-1][2*j+1]),
                    .s_axis_b_tlast(stage_last[stage-1][2*j+1]),
                    .s_axis_b_tdata(stage_data[stage-1][2*j+1]),
                    .m_axis_result_tvalid(stage_valid[stage][j]),
                    .m_axis_result_tlast(stage_last[stage][j]),
                    .m_axis_result_tdata(stage_data[stage][j])
                );
            end
        end
    endgenerate
    
    assign out = stage_data[ADD_STAGES][0];
    assign out_valid = stage_valid[ADD_STAGES][0];
    assign out_last = stage_last[ADD_STAGES][0];

endmodule