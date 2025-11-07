module FM_top #(
    parameter  hidden_states = 16,
    parameter  sequence_length = 2,
    parameter  bram_depth = hidden_states * sequence_length,
    parameter  bitwidth = 16,
    parameter  std_N = 8,
    parameter  mean_N = 8
) (
    input wire clk,
    input wire rstn,
    input wire mode,

    input wire  en,

    input wire [mean_N*bitwidth-1:0] x1,
    output wire [$clog2(bram_depth/mean_N)-1:0] x1_addr,

    input wire [mean_N*bitwidth-1:0] x2,
    output wire [$clog2(bram_depth/mean_N)-1:0] x2_addr,

    input wire [mean_N*bitwidth-1:0] weight,
    input wire [mean_N*bitwidth-1:0] bias,
    output wire [$clog2(bram_depth/mean_N)-1:0] param_addr,

    input wire [bitwidth-1:0] H_param,
    input wire [bitwidth-1:0] C_param,
    input wire [bitwidth-1:0] E_param,

    output wire [mean_N*bitwidth-1:0] normal_out,
    output wire normal_out_valid,
    output wire normal_out_last,

    output wire [bitwidth-1:0]   one_variance,
    output wire [mean_N*bitwidth-1:0] FM_out,
    output wire FM_out_valid,
    output wire FM_out_last,

    output wire [std_N-1:0] max_index,
    output wire [std_N-1:0] min_index,
    output wire index_valid,
    output wire index_last,

    output wire [bitwidth-1:0] mean_out_r_t,
    output wire [bitwidth-1:0] variance_out_r_t,
    output wire [bitwidth-1:0] mean_variance_out_r_t,
    output wire [bitwidth-1:0] one_variance_out_r_t
);

assign one_variance = one_variance_out_r;

assign mean_variance_out_r_t = mean_variance_out_r;
assign one_variance_out_r_t = one_variance_out_r;

localparam  in1_max_add = hidden_states/mean_N;
reg vector1_addr_cnt_en;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        vector1_addr_cnt_en <= 0;
    end
    else if (en) begin
        vector1_addr_cnt_en <= 1;
    end
    else if (first_stage_last && vector1_addr_cnt_switch) begin
        vector1_addr_cnt_en <= 1;
    end
    else if (vector1_addr_cnt == in1_max_add - 1) begin  
        vector1_addr_cnt_en <= 0;
    end
end

reg [$clog2(bram_depth/mean_N)-1:0] vector1_addr_cnt;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        vector1_addr_cnt <= 0;
    end
    else if (vector1_addr_cnt_en) begin
        if (vector1_addr_cnt == in1_max_add - 1) begin
            vector1_addr_cnt <= 0;
        end
        else begin
            vector1_addr_cnt <= vector1_addr_cnt + 1;
        end
    end
    else begin
        vector1_addr_cnt <= 0;
    end
end

assign last_seq = (base_addr == (sequence_length-1)*in1_max_add);
assign x1_addr = base_addr + vector1_addr_cnt;
reg [$clog2(bram_depth/mean_N)-1:0] base_addr;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        base_addr <= 0;
    end
    else if (vector1_addr_cnt == in1_max_add - 1 && ~last_seq) begin
        base_addr <= base_addr + in1_max_add;
    end
    else if(vector1_addr_cnt == in1_max_add - 1 && last_seq) begin
        base_addr <= 0;
    end
end

reg vector1_addr_cnt_switch;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        vector1_addr_cnt_switch <= 0;
    end
    else if (en) begin
        vector1_addr_cnt_switch <= 1;
    end
    else if(vector1_addr_cnt == in1_max_add - 1 && last_seq) begin
        vector1_addr_cnt_switch <= 0;
    end
end

reg  x1_valid;
reg  x1_last;
wire x1_last_w;
assign x1_last_w = (vector1_addr_cnt == in1_max_add - 1) ? 1 : 0;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        x1_valid <= 0;
        x1_last <= 0;
    end
    else if (vector1_addr_cnt_en) begin
        x1_valid <= 1;
        x1_last <= x1_last_w;
    end
    else begin
        x1_valid <= 0;
        x1_last <= 0;
    end
end

wire [bitwidth-1:0] mean_variance_out;
wire [bitwidth-1:0] one_variance_out;
wire first_stage_valid;
wire first_stage_last;
FM_first_stage #(
    .bitwidth(bitwidth),
    .mean_N(mean_N),
    .std_N(std_N)
) u_FM_first_stage (
    .clk(clk),
    .rstn(rstn),
    .mode(mode),
    .x_mean(x1),
    .x_mean_valid(x1_valid),
    .x_mean_last(x1_last),

    .x_std(x1),
    .x_std_valid(x1_valid),
    .x_std_last(x1_last),

    .H_param(H_param),
    .C_param(C_param),
    .E_param(E_param),

    .max_index(max_index),
    .min_index(min_index),
    .index_valid(index_valid),
    .index_last(index_last),

    .mean_variance_out(mean_variance_out),
    .one_variance_out(one_variance_out),
    .out_valid(first_stage_valid),
    .out_last(first_stage_last)
);

reg [bitwidth-1:0] mean_variance_out_r;
reg [bitwidth-1:0] one_variance_out_r;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        mean_variance_out_r <= 0;
        one_variance_out_r <= 0;
    end
    else if (first_stage_last) begin
        mean_variance_out_r <= mean_variance_out;
        one_variance_out_r <= one_variance_out;
    end
end

//in2 addr control
localparam  in2_max_add = hidden_states/mean_N;
reg vector2_addr_cnt_en;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        vector2_addr_cnt_en <= 0;
    end
    else if (first_stage_last && vector2_addr_cnt_switch) begin
        vector2_addr_cnt_en <= 1;
    end
    else if (vector2_addr_cnt == in2_max_add - 1) begin  
        vector2_addr_cnt_en <= 0;
    end
end

reg [$clog2(bram_depth/mean_N)-1:0] vector2_addr_cnt;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        vector2_addr_cnt <= 0;
    end
    else if (vector2_addr_cnt_en) begin
        if (vector2_addr_cnt == in2_max_add - 1) begin
            vector2_addr_cnt <= 0;
        end
        else begin
            vector2_addr_cnt <= vector2_addr_cnt + 1;
        end
    end
    else begin
        vector2_addr_cnt <= 0;
    end
end

wire last_seq_second_stage;
assign last_seq_second_stage = (base_addr_second_stage == (sequence_length-1)*in2_max_add);
assign x2_addr = base_addr_second_stage + vector2_addr_cnt;
reg [$clog2(bram_depth/mean_N)-1:0] base_addr_second_stage;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        base_addr_second_stage <= 0;
    end
    else if (vector2_addr_cnt == in2_max_add - 1 && ~last_seq_second_stage) begin
        base_addr_second_stage <= base_addr_second_stage + in2_max_add;
    end
    else if(vector2_addr_cnt == in2_max_add - 1 && last_seq_second_stage) begin
        base_addr_second_stage <= 0;
    end
end

reg vector2_addr_cnt_switch;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        vector2_addr_cnt_switch <= 0;
    end
    else if (en) begin
        vector2_addr_cnt_switch <= 1;
    end
    else if(vector2_addr_cnt == in2_max_add - 1 && last_seq_second_stage) begin
        vector2_addr_cnt_switch <= 0;
    end
end


reg  x2_valid;
reg  x2_last;
wire x2_last_w;
assign x2_last_w = (vector2_addr_cnt == in2_max_add - 1) ? 1 : 0;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        x2_valid <= 0;
        x2_last <= 0;
    end
    else if (vector2_addr_cnt_en) begin
        x2_valid <= 1;
        x2_last <= x2_last_w;
    end
    else begin
        x2_valid <= 0;
        x2_last <= 0;
    end
end

//param addr control
reg [$clog2(bram_depth/mean_N)-1:0] param_addr_cnt;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        param_addr_cnt <= 0;
    end
    else if (normal_out_valid && param_addr_cnt_switch) begin
        if (param_addr_cnt == param_max_add - 1) begin
            param_addr_cnt <= 0;
        end
        else begin
            param_addr_cnt <= param_addr_cnt + 1;
        end
    end
    else begin
        param_addr_cnt <= 0;
    end
end

wire last_seq_param;
localparam  param_max_add = hidden_states/mean_N;
assign last_seq_param = (base_addr_param == (sequence_length-1)*param_max_add);
assign param_addr = base_addr_param + param_addr_cnt;
reg [$clog2(bram_depth/mean_N)-1:0] base_addr_param;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        base_addr_param <= 0;
    end
    else if (param_addr_cnt == param_max_add - 1 && ~last_seq_param) begin
        base_addr_param <= base_addr_param + param_max_add;
    end
    else if(param_addr_cnt == param_max_add - 1 && last_seq_param) begin
        base_addr_param <= 0;
    end
end

reg param_addr_cnt_switch;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        param_addr_cnt_switch <= 0;
    end
    else if (en) begin
        param_addr_cnt_switch <= 1;
    end
    else if(param_addr_cnt == param_max_add - 1 && last_seq_param) begin
        param_addr_cnt_switch <= 0;
    end
end

reg  param_valid;
reg  param_last;
wire param_last_w;
assign param_last_w = (param_addr_cnt == param_max_add - 1) ? 1 : 0;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        param_valid <= 0;
        param_last <= 0;
    end
    else if (normal_out_valid) begin
        param_valid <= 1;
        param_last <= param_last_w;
    end
    else begin
        param_valid <= 0;
        param_last <= 0;
    end
end

FM_last_stage #(
    .bitwidth(bitwidth),
    .mean_N(mean_N)
) u_FM_last_stage (
    .clk(clk),
    .rstn(rstn),
    .mode(mode),

    .x(x2),
    .x_valid(x2_valid),
    .x_last(x2_last),
    .mean_variance(mean_variance_out_r),
    .one_variance(one_variance_out_r),
    .normal_out(normal_out),
    .normal_out_valid(normal_out_valid),
    .normal_out_last(normal_out_last),

    .weight(weight),
    .bias(bias),
    .param_valid(param_valid),
    .param_last(param_last),
    .out(FM_out),
    .out_valid(FM_out_valid),
    .out_last(FM_out_last)
);

endmodule