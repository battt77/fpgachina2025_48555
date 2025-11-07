module BM_top #(
    parameter  hidden_p = 16, 
    parameter  sequence_length = 2, 
    parameter  bram_depth = hidden_p * sequence_length, 
    parameter  bitwidth = 16, 
    parameter  N = 8
)(
    input wire clk,
    input wire rstn,
    input wire en,
    input wire mode, //0: hfln; 1: rms

    input wire [N*bitwidth-1:0] dz1,
    input wire [N*bitwidth-1:0] y,
    output wire [$clog2(bram_depth/N)-1:0] dz_y_addr,

    input wire [N*bitwidth-1:0] dz2,
    input wire [N-1:0] max_index,
    input wire [N-1:0] min_index,
    input wire [N*bitwidth-1:0] weight,
    input wire [N*bitwidth-1:0] y2,
    output wire [$clog2(bram_depth/N)-1:0] y2_addr,
    output wire [$clog2(bram_depth/N)-1:0] second_stage_addr,
   
    input wire [bitwidth-1:0] H_param,
    input wire [bitwidth-1:0] C_param,
    input wire [bitwidth-1:0] variance_rec,

    output wire [N*bitwidth-1:0] gradx_out,
    output wire gradx_out_valid,
    output wire gradx_out_last
);
/////************addr for dz1_y************/////
localparam  dz1_max_add = hidden_p/N;
reg dz1_addr_cnt_en;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dz1_addr_cnt_en <= 0;
    end
    else if (en) begin
        dz1_addr_cnt_en <= 1;
    end
    else if (first_stage_last && dz1_addr_cnt_switch) begin
        dz1_addr_cnt_en <= 1;
    end
    else if (dz1_addr_cnt == dz1_max_add - 1) begin  
        dz1_addr_cnt_en <= 0;
    end
end

reg [$clog2(bram_depth/N)-1:0] dz1_addr_cnt;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dz1_addr_cnt <= 0;
    end
    else if (dz1_addr_cnt_en) begin
        if (dz1_addr_cnt == dz1_max_add - 1) begin
            dz1_addr_cnt <= 0;
        end
        else begin
            dz1_addr_cnt <= dz1_addr_cnt + 1;
        end
    end
    else begin
        dz1_addr_cnt <= 0;
    end
end

wire last_seq_dz1;
assign last_seq_dz1 = (base_addr_dz1 == (sequence_length-1)*dz1_max_add);
assign dz_y_addr = base_addr_dz1 + dz1_addr_cnt;
reg [$clog2(bram_depth/N)-1:0] base_addr_dz1;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        base_addr_dz1 <= 0;
    end
    else if (dz1_addr_cnt == dz1_max_add - 1 && ~last_seq_dz1) begin
        base_addr_dz1 <= base_addr_dz1 + dz1_max_add;
    end
    else if(dz1_addr_cnt == dz1_max_add - 1 && last_seq_dz1) begin
        base_addr_dz1 <= 0;
    end
end

reg dz1_addr_cnt_switch;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dz1_addr_cnt_switch <= 0;
    end
    else if (en) begin
        dz1_addr_cnt_switch <= 1;
    end
    else if(dz1_addr_cnt == dz1_max_add - 1 && last_seq_dz1) begin
        dz1_addr_cnt_switch <= 0;
    end
end

reg  dz1_y_valid;
reg  dz1_y_last;
wire dz1_y_last_w;
assign dz1_y_last_w = (dz1_addr_cnt == dz1_max_add - 1) ? 1 : 0;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dz1_y_valid <= 0;
        dz1_y_last <= 0;
    end
    else if (dz1_addr_cnt_en) begin
        dz1_y_valid <= 1;
        dz1_y_last <= dz1_y_last_w;
    end
    else begin
        dz1_y_valid <= 0;
        dz1_y_last <= 0;
    end
end
//////////********addr for dz1_y end

/////************addr for y2************/////
localparam  y2_max_add = hidden_p/N;
reg y2_addr_cnt_en;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        y2_addr_cnt_en <= 0;
    end
    else if (gradw_out_last && y2_addr_cnt_switch) begin
        y2_addr_cnt_en <= 1;
    end
    else if (y2_addr_cnt == y2_max_add - 1) begin  
        y2_addr_cnt_en <= 0;
    end
end

reg [$clog2(bram_depth/N)-1:0] y2_addr_cnt;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        y2_addr_cnt <= 0;
    end
    else if (y2_addr_cnt_en) begin
        if (y2_addr_cnt == y2_max_add - 1) begin
            y2_addr_cnt <= 0;
        end
        else begin
            y2_addr_cnt <= y2_addr_cnt + 1;
        end
    end
    else begin
        y2_addr_cnt <= 0;
    end
end

wire last_seq_y2;
assign last_seq_y2 = (base_addr_y2 == (sequence_length-1)*y2_max_add);
assign y2_addr = base_addr_y2 + y2_addr_cnt;
reg [$clog2(bram_depth/N)-1:0] base_addr_y2;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        base_addr_y2 <= 0;
    end
    else if (y2_addr_cnt == y2_max_add - 1 && ~last_seq_y2) begin
        base_addr_y2 <= base_addr_y2 + y2_max_add;
    end
    else if(y2_addr_cnt == y2_max_add - 1 && last_seq_y2) begin
        base_addr_y2 <= 0;
    end
end

reg y2_addr_cnt_switch;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        y2_addr_cnt_switch <= 0;
    end
    else if (en) begin
        y2_addr_cnt_switch <= 1;
    end
    else if(y2_addr_cnt == y2_max_add - 1 && last_seq_y2) begin
        y2_addr_cnt_switch <= 0;
    end
end

reg  y2_valid;
reg  y2_last;
wire y2_last_w;
assign y2_last_w = (y2_addr_cnt == y2_max_add - 1) ? 1 : 0;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        y2_valid <= 0;
        y2_last <= 0;
    end
    else if (y2_addr_cnt_en) begin
        y2_valid <= 1;
        y2_last <= y2_last_w;
    end
    else begin
        y2_valid <= 0;
        y2_last <= 0;
    end
end
//////////********addr for y2 end

wire [bitwidth-1:0] gradb_add_gradg;
wire [bitwidth-1:0] gradb_sub_gradg;
wire [bitwidth-1:0] gradb;
wire [N*bitwidth-1:0] gradw_y2_mult;
wire first_stage_valid;
wire first_stage_last;
wire dz2start;

BM_first_stage #(
    .bitwidth(bitwidth),
    .N(N),
    .hidden_num(hidden_p)
) u_BM_first_stage (
    .clk(clk),
    .rstn(rstn),
    .gradz(dz1),
    .y1(y),
    .y2(y2),
    .in_valid(dz1_y_valid),
    .in_last(dz1_y_last),
    .y2_in_valid(y2_valid),
    .y2_in_last(y2_last),
    .mode(mode),
    .H_param(H_param),
    .C_param(C_param),
    .gradw_out_last(gradw_out_last),
    .gradw_y2_mult_out(gradw_y2_mult),
    .gradb_add_gradg(gradb_add_gradg),
    .gradb_sub_gradg(gradb_sub_gradg),
    .gradb(gradb),
    .out_valid(first_stage_valid),
    .out_last(first_stage_last),
    .dz2start(dz2start)
);


reg [bitwidth-1:0] gradb_add_gradg_r;
reg [bitwidth-1:0] gradb_sub_gradg_r;
reg [bitwidth-1:0] gradb_r;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        gradb_add_gradg_r <= 0;
        gradb_sub_gradg_r <= 0;
        gradb_r <= 0;
    end
    else if (first_stage_last) begin
        gradb_add_gradg_r <= gradb_add_gradg;
        gradb_sub_gradg_r <= gradb_sub_gradg;
        gradb_r <= gradb;
    end
end

///delay circuit for gradw_y2_mult (2 cycles delay)
reg [N*bitwidth-1:0] gradw_y2_mult_d1;
reg [N*bitwidth-1:0] gradw_y2_mult_d2;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        gradw_y2_mult_d1 <= 0;
        gradw_y2_mult_d2 <= 0;
    end
    else begin
        // First delay stage
        gradw_y2_mult_d1 <= gradw_y2_mult;
        // Second delay stage
        gradw_y2_mult_d2 <= gradw_y2_mult_d1;
    end
end


localparam  dz2_max_add = hidden_p/N;
reg dz2_addr_cnt_en;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dz2_addr_cnt_en <= 0;
    end
    else if (dz2start && dz2_addr_cnt_switch) begin
        dz2_addr_cnt_en <= 1;
    end
    else if (dz2_addr_cnt == dz2_max_add - 1) begin  
        dz2_addr_cnt_en <= 0;
    end
end

reg [$clog2(bram_depth/N)-1:0] dz2_addr_cnt;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dz2_addr_cnt <= 0;
    end
    else if (dz2_addr_cnt_en) begin
        if (dz2_addr_cnt == dz2_max_add - 1) begin
            dz2_addr_cnt <= 0;
        end
        else begin
            dz2_addr_cnt <= dz2_addr_cnt + 1;
        end
    end
    else begin
        dz2_addr_cnt <= 0;
    end
end

wire last_seq_dz2;
assign last_seq_dz2 = (base_addr_dz2 == (sequence_length-1)*dz2_max_add);
assign second_stage_addr = base_addr_dz2 + dz2_addr_cnt;
reg [$clog2(bram_depth/N)-1:0] base_addr_dz2;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        base_addr_dz2 <= 0;
    end
    else if (dz2_addr_cnt == dz2_max_add - 1 && ~last_seq_dz2) begin
        base_addr_dz2 <= base_addr_dz2 + dz2_max_add;
    end
    else if(dz2_addr_cnt == dz2_max_add - 1 && last_seq_dz2) begin
        base_addr_dz2 <= 0;
    end
end

reg dz2_addr_cnt_switch;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dz2_addr_cnt_switch <= 0;
    end
    else if (en) begin
        dz2_addr_cnt_switch <= 1;
    end
    else if(dz2_addr_cnt == dz2_max_add - 1 && last_seq_dz2) begin
        dz2_addr_cnt_switch <= 0;
    end
end

reg  second_stage_valid;
reg  second_stage_last;
wire second_stage_last_w;
assign second_stage_last_w = (dz2_addr_cnt == dz2_max_add - 1) ? 1 : 0;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        second_stage_valid <= 0;
        second_stage_last <= 0;
    end
    else if (dz2_addr_cnt_en) begin
        second_stage_valid <= 1;
        second_stage_last <= second_stage_last_w;
    end
    else begin
        second_stage_valid <= 0;
        second_stage_last <= 0;
    end
end

BM_last_stage #(
    .bitwidth(bitwidth),
    .N(N),
    .hidden_num(hidden_p)
) u_BM_last_stage (
    .clk(clk),
    .rstn(rstn),

    .gradb_add_gradg(gradb_add_gradg_r),
    .gradb_sub_gradg(gradb_sub_gradg_r),
    .gradb(gradb_r),
    .gradw_y2_mult(gradw_y2_mult_d2),

    .dz(dz2),
    .mode(mode),
    .max_index(max_index),
    .min_index(min_index),
    .in_valid(second_stage_valid),
    .in_last(second_stage_last),

    .weight(weight), ///m^(-1/2) for rms
    .variance_rec(variance_rec),
    .param_valid(second_stage_valid),
    .param_last(second_stage_last),

    .gradx_out(gradx_out),
    .gradx_out_valid(gradx_out_valid),
    .gradx_out_last(gradx_out_last)
);

endmodule


