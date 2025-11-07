module vector_compare_tree #(
    parameter bitwidth = 16,
    parameter N = 8
) (
    input wire clk,
    input wire rstn,
    
    input wire [N*bitwidth-1:0]  in,
    input wire in_valid,
    input wire in_last,

    output reg [bitwidth-1:0]   max_out,
    output wire [N-1:0]   max_index,
    output reg [bitwidth-1:0]   min_out,
    output wire [N-1:0]   min_index,
    output reg out_valid,
    output reg out_last
);
    localparam compare_ip_clk = 2; 
    localparam ADD_STAGES = $clog2(N) - 1;  
    wire [7:0] stage_compare_flag_max [0:ADD_STAGES][0:N-1];
    wire [7:0] stage_compare_flag_min [0:ADD_STAGES][0:N-1];
    wire [bitwidth-1:0] stage_max_data [0:ADD_STAGES][0:N/2-1];
    wire [bitwidth-1:0] stage_min_data [0:ADD_STAGES][0:N/2-1];

    wire [N-1:0] stage_valid [0:ADD_STAGES];
    wire [N-1:0] stage_valid_min [0:ADD_STAGES];
    wire [N-1:0] stage_last  [0:ADD_STAGES];
    wire [N-1:0] stage_last_min  [0:ADD_STAGES];

    reg  [N-1:0] max_index_reg [0:ADD_STAGES];
    reg  [N-1:0] min_index_reg [0:ADD_STAGES];
    
    // Module-level delay registers for stage 1+ (to avoid hierarchical reference in always block)
    reg  [N-1:0] max_index_delayed [0:ADD_STAGES][0:compare_ip_clk-2];
    reg  [N-1:0] min_index_delayed [0:ADD_STAGES][0:compare_ip_clk-2];

    // Stage 0: Index register update (no delay needed, direct from compare results)
    integer idx0;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            max_index_reg[0] <= 0;
            min_index_reg[0] <= 0;
        end
        else if (out_last) begin
            max_index_reg[0] <= 0;
            min_index_reg[0] <= 0;
        end
        else begin
            for (idx0 = 0; idx0 < N/2; idx0 = idx0 + 1) begin
                if (stage_valid[0][idx0]) begin
                    case (stage_compare_flag_max[0][idx0][2:0])
                    3'b001:begin
                        max_index_reg[0][2*idx0 +: 2] <= 2'b11;
                        min_index_reg[0][2*idx0 +: 2] <= 2'b11;
                    end
                    3'b010:begin
                        max_index_reg[0][2*idx0 +: 2] <= 2'b10;
                        min_index_reg[0][2*idx0 +: 2] <= 2'b01;
                    end 
                    3'b100:begin
                        max_index_reg[0][2*idx0 +: 2] <= 2'b01;
                        min_index_reg[0][2*idx0 +: 2] <= 2'b10;
                    end
                    default: begin
                        max_index_reg[0][2*idx0 +: 2] <= 2'b00;
                        min_index_reg[0][2*idx0 +: 2] <= 2'b00;
                    end
                    endcase
                end
            end
        end
    end
    
    // Stage 1+: Delay chains for index propagation
    integer st, k_delay;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (st = 1; st <= ADD_STAGES; st = st + 1) begin
                for (k_delay = 0; k_delay < compare_ip_clk-1; k_delay = k_delay + 1) begin
                    max_index_delayed[st][k_delay] <= 0;
                    min_index_delayed[st][k_delay] <= 0;
                end
            end
        end
        else begin
            for (st = 1; st <= ADD_STAGES; st = st + 1) begin
                max_index_delayed[st][0] <= max_index_reg[st-1];
                min_index_delayed[st][0] <= min_index_reg[st-1];
                for (k_delay = 1; k_delay < compare_ip_clk-1; k_delay = k_delay + 1) begin
                    max_index_delayed[st][k_delay] <= max_index_delayed[st][k_delay-1];
                    min_index_delayed[st][k_delay] <= min_index_delayed[st][k_delay-1];
                end
            end
        end
    end

    genvar i,j,k0;
    generate
        localparam  compare_ip_clk_stage0= compare_ip_clk;
        for (i = 0; i < N/2; i = i + 1) begin: stage0
            reg [bitwidth-1:0] a_delay [0:compare_ip_clk_stage0-1];
            reg [bitwidth-1:0] b_delay [0:compare_ip_clk_stage0-1];

            integer k0;
            always @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    for (k0=0;k0<compare_ip_clk_stage0;k0=k0+1) begin
                        a_delay[k0]<= 0;
                        b_delay[k0]<= 0;
                    end
                end 
                else begin
                    a_delay[0] <= in[(2*i)*bitwidth +: bitwidth];
                    b_delay[0] <= in[(2*i+1)*bitwidth +: bitwidth];
                    for (k0=1;k0<compare_ip_clk_stage0;k0=k0+1) begin
                        a_delay[k0]<= a_delay[k0-1];
                        b_delay[k0]<= b_delay[k0-1];
                    end
                end
            end
            
            floating_compare u_compare_0 (
                .aclk(clk),
                .aresetn(rstn),
                .s_axis_a_tvalid(in_valid),
                .s_axis_a_tlast(in_last),
                .s_axis_a_tdata(in[(2*i)*bitwidth +: bitwidth]),
                .s_axis_b_tvalid(in_valid),
                .s_axis_b_tlast(in_last),
                .s_axis_b_tdata(in[(2*i+1)*bitwidth +: bitwidth]),
                .m_axis_result_tvalid(stage_valid[0][i]),
                .m_axis_result_tlast(stage_last[0][i]),
                .m_axis_result_tdata(stage_compare_flag_max[0][i])
            );

            assign stage_valid_min[0][i] = stage_valid[0][i];
            assign stage_last_min[0][i] = stage_last[0][i];
            // stage_compare_flag[0][i][0]: a == b
            // stage_compare_flag[0][i][1]: a < b  
            // stage_compare_flag[0][i][2]: a > b
            
            // 最大值选择：a>=b 时选 a，否则选 b
            assign stage_max_data[0][i] = (stage_compare_flag_max[0][i][0] | stage_compare_flag_max[0][i][2]) ? 
                                          a_delay[compare_ip_clk_stage0-1] : 
                                          b_delay[compare_ip_clk_stage0-1];
            
            // 最小值选择：a<=b 时选 a，否则选 b
            assign stage_min_data[0][i] = (stage_compare_flag_max[0][i][0] | stage_compare_flag_max[0][i][1]) ? 
                                          a_delay[compare_ip_clk_stage0-1] : 
                                          b_delay[compare_ip_clk_stage0-1];
            
            // Index update logic has been moved outside generate (see line 36-69)
        end
    endgenerate

    // Stage 1: Index register update (for N=8, 2 nodes, each handles 4 bits)
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            max_index_reg[1] <= 0;
            min_index_reg[1] <= 0;
        end
        else if (out_last) begin
            max_index_reg[1] <= 0;
            min_index_reg[1] <= 0;
        end
        else begin
            // Node 0: handles bits [3:0]
            if (stage_valid_min[1][0] & stage_valid[1][0]) begin
                case (stage_compare_flag_max[1][0][2:0])
                3'b001: max_index_reg[1][3:0] <= {max_index_delayed[1][compare_ip_clk-2][3:2], max_index_delayed[1][compare_ip_clk-2][1:0]};
                3'b010: max_index_reg[1][3:0] <= {max_index_delayed[1][compare_ip_clk-2][3:2], 2'b00};
                3'b100: max_index_reg[1][3:0] <= {2'b00, max_index_delayed[1][compare_ip_clk-2][1:0]};
                default: max_index_reg[1][3:0] <= 0;
                endcase
                case (stage_compare_flag_min[1][0][2:0])
                3'b001: min_index_reg[1][3:0] <= {min_index_delayed[1][compare_ip_clk-2][3:2], min_index_delayed[1][compare_ip_clk-2][1:0]};
                3'b010: min_index_reg[1][3:0] <= {2'b00, min_index_delayed[1][compare_ip_clk-2][1:0]};
                3'b100: min_index_reg[1][3:0] <= {min_index_delayed[1][compare_ip_clk-2][3:2], 2'b00};
                default: min_index_reg[1][3:0] <= 0;
                endcase
            end
            // Node 1: handles bits [7:4]
            if (stage_valid_min[1][1] & stage_valid[1][1]) begin
                case (stage_compare_flag_max[1][1][2:0])
                3'b001: max_index_reg[1][7:4] <= {max_index_delayed[1][compare_ip_clk-2][7:6], max_index_delayed[1][compare_ip_clk-2][5:4]};
                3'b010: max_index_reg[1][7:4] <= {max_index_delayed[1][compare_ip_clk-2][7:6], 2'b00};
                3'b100: max_index_reg[1][7:4] <= {2'b00, max_index_delayed[1][compare_ip_clk-2][5:4]};
                default: max_index_reg[1][7:4] <= 0;
                endcase
                case (stage_compare_flag_min[1][1][2:0])
                3'b001: min_index_reg[1][7:4] <= {min_index_delayed[1][compare_ip_clk-2][7:6], min_index_delayed[1][compare_ip_clk-2][5:4]};
                3'b010: min_index_reg[1][7:4] <= {2'b00, min_index_delayed[1][compare_ip_clk-2][5:4]};
                3'b100: min_index_reg[1][7:4] <= {min_index_delayed[1][compare_ip_clk-2][7:6], 2'b00};
                default: min_index_reg[1][7:4] <= 0;
                endcase
            end
        end
    end
    
    // Stage 2: Index register update (for N=8, 1 node, handles all 8 bits)
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            max_index_reg[2] <= 0;
            min_index_reg[2] <= 0;
        end
        else if (out_last) begin
            max_index_reg[2] <= 0;
            min_index_reg[2] <= 0;
        end
        else begin
            if (stage_valid_min[2][0] & stage_valid[2][0]) begin
                case (stage_compare_flag_max[2][0][2:0])
                3'b001: max_index_reg[2] <= {max_index_delayed[2][compare_ip_clk-2][7:4], max_index_delayed[2][compare_ip_clk-2][3:0]};
                3'b010: max_index_reg[2] <= {max_index_delayed[2][compare_ip_clk-2][7:4], 4'b0000};
                3'b100: max_index_reg[2] <= {4'b0000, max_index_delayed[2][compare_ip_clk-2][3:0]};
                default: max_index_reg[2] <= 0;
                endcase
                case (stage_compare_flag_min[2][0][2:0])
                3'b001: min_index_reg[2] <= {min_index_delayed[2][compare_ip_clk-2][7:4], min_index_delayed[2][compare_ip_clk-2][3:0]};
                3'b010: min_index_reg[2] <= {4'b0000, min_index_delayed[2][compare_ip_clk-2][3:0]};
                3'b100: min_index_reg[2] <= {min_index_delayed[2][compare_ip_clk-2][7:4], 4'b0000};
                default: min_index_reg[2] <= 0;
                endcase
            end
        end
    end

    genvar stage;
    generate
        for (stage = 1; stage <= ADD_STAGES; stage = stage + 1) begin: add_tree_stage
            localparam NUM_NODES = N >> 1 >> stage; 

            for (j = 0; j < NUM_NODES; j = j + 1) begin: add_node
                // Max路径的延迟线
                reg [bitwidth-1:0] a_delay_max [0:compare_ip_clk-1];
                reg [bitwidth-1:0] b_delay_max [0:compare_ip_clk-1];
                // Min路径的延迟线
                reg [bitwidth-1:0] a_delay_min [0:compare_ip_clk-1];
                reg [bitwidth-1:0] b_delay_min [0:compare_ip_clk-1];

                integer k1;
                always @(posedge clk or negedge rstn) begin
                    if (!rstn) begin
                        for (k1=0; k1<compare_ip_clk; k1=k1+1) begin
                            a_delay_max[k1] <= 0;
                            b_delay_max[k1] <= 0;
                            a_delay_min[k1] <= 0;
                            b_delay_min[k1] <= 0;
                        end
                    end 
                    else begin
                        // Max路径延迟
                        a_delay_max[0] <= stage_max_data[stage-1][2*j];
                        b_delay_max[0] <= stage_max_data[stage-1][2*j+1];
                        // Min路径延迟
                        a_delay_min[0] <= stage_min_data[stage-1][2*j];
                        b_delay_min[0] <= stage_min_data[stage-1][2*j+1];
                        
                        for (k1=1; k1<compare_ip_clk; k1=k1+1) begin
                            a_delay_max[k1] <= a_delay_max[k1-1];
                            b_delay_max[k1] <= b_delay_max[k1-1];
                            a_delay_min[k1] <= a_delay_min[k1-1];
                            b_delay_min[k1] <= b_delay_min[k1-1];
                        end
                    end
                end
                
                // Index delay logic has been moved to module level (see line 75-96)



                // 用Max路径的数据进行比较
                floating_compare u_compare_max_node (
                    .aclk(clk),
                    .aresetn(rstn),
                    .s_axis_a_tvalid(stage_valid[stage-1][2*j]),
                    .s_axis_a_tlast(stage_last[stage-1][2*j]),
                    .s_axis_a_tdata(stage_max_data[stage-1][2*j]),
                    .s_axis_b_tvalid(stage_valid[stage-1][2*j+1]),
                    .s_axis_b_tlast(stage_last[stage-1][2*j+1]),
                    .s_axis_b_tdata(stage_max_data[stage-1][2*j+1]),
                    .m_axis_result_tvalid(stage_valid[stage][j]),
                    .m_axis_result_tlast(stage_last[stage][j]),
                    .m_axis_result_tdata(stage_compare_flag_max[stage][j])
                );

                floating_compare u_compare_min_node (
                    .aclk(clk),
                    .aresetn(rstn),
                    .s_axis_a_tvalid(stage_valid_min[stage-1][2*j]),
                    .s_axis_a_tlast(stage_last_min[stage-1][2*j]),
                    .s_axis_a_tdata(stage_min_data[stage-1][2*j]),
                    .s_axis_b_tvalid(stage_valid_min[stage-1][2*j+1]),
                    .s_axis_b_tlast(stage_last_min[stage-1][2*j+1]),
                    .s_axis_b_tdata(stage_min_data[stage-1][2*j+1]),
                    .m_axis_result_tvalid(stage_valid_min[stage][j]),
                    .m_axis_result_tlast(stage_last_min[stage][j]),
                    .m_axis_result_tdata(stage_compare_flag_min[stage][j])
                );
                
                
                // 最大值路径：比较两个max，选择较大的 (a>=b 时选a)
                assign stage_max_data[stage][j] = (stage_compare_flag_max[stage][j][0] | stage_compare_flag_max[stage][j][2]) ? 
                                                  a_delay_max[compare_ip_clk-1] : 
                                                  b_delay_max[compare_ip_clk-1];
                
                // 最小值路径：比较两个min，选择较小的 (a<=b 时选a)
                assign stage_min_data[stage][j] = (stage_compare_flag_min[stage][j][0] | stage_compare_flag_min[stage][j][1]) ? 
                                                  a_delay_min[compare_ip_clk-1] : 
                                                  b_delay_min[compare_ip_clk-1];
                
                // Index update logic has been moved outside generate (see line 157-210)
            end
        end
    endgenerate
    

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            max_out <= 0;
            min_out <= 0;
            out_valid <= 0;
            out_last <= 0;
        end
        else begin
            max_out <= stage_max_data[ADD_STAGES][0];
            min_out <= stage_min_data[ADD_STAGES][0];
            out_valid <= stage_valid[ADD_STAGES][0] & stage_valid_min[ADD_STAGES][0];
            out_last <= stage_last[ADD_STAGES][0] & stage_last_min[ADD_STAGES][0];
        end
    end

    assign max_index = max_index_reg[ADD_STAGES];
    assign min_index = min_index_reg[ADD_STAGES];

endmodule