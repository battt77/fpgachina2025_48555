module BM_param_top #(
    parameter bitwidth = 16,
    parameter sequence_length = 2,
    parameter hidden_dim = 8,
    parameter addr_bitwidth = $clog2(hidden_dim) 
) (
    input wire clk,
    input wire rstn,

    input wire en,

    input wire [sequence_length*bitwidth-1:0] gradz,
    input wire [sequence_length*bitwidth-1:0] y1,
    output reg [addr_bitwidth-1:0]           addr,

    output wire [bitwidth-1:0] gradg_out,
    output wire gradg_out_valid,
    output wire gradg_out_last,

    output wire [bitwidth-1:0] gradb_out,
    output wire gradb_out_valid,
    output wire gradb_out_last
);

localparam N = sequence_length;

reg addr_cnt_en;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        addr_cnt_en <= 0;
    end
    else if (en) begin
        addr_cnt_en <= 1;
    end
    else if (addr==hidden_dim-1) begin
        addr_cnt_en <= 0;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        addr <= 0;
    end 
    else if (addr_cnt_en) begin
        addr <= addr + 1;
    end
    else begin
        addr <= 0;
    end
end

reg in_valid;
wire in_last_w;
reg in_last;
assign in_last_w = (addr == hidden_dim - 1);
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        in_valid <= 0;
        in_last <= 0;
    end
    else if (addr_cnt_en) begin
        in_valid <= 1;
        in_last <= in_last_w;
    end
    else begin
        in_valid <= 0;
        in_last <= 0;
    end
end

reduction_tree #(
    .bitwidth(bitwidth),
    .N(N)
) u_gradz_sum (
    .clk(clk),
    .rstn(rstn),
    .in(gradz),
    .in_valid(in_valid),
    .in_last(in_last),
    .out(gradb_out),
    .out_valid(gradb_out_valid),
    .out_last(gradb_out_last)
);

wire [bitwidth-1:0] gradw_sum;
wire gradw_valid;
wire gradw_last;
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
    .out(gradg_out),
    .out_valid(gradg_out_valid),
    .out_last(gradg_out_last)
);

endmodule