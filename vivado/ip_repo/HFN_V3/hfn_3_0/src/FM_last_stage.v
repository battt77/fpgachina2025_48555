module FM_last_stage #(
    parameter bitwidth = 16,
    parameter mean_N = 8
) (
    input wire clk,
    input wire rstn,
    input wire mode,

    input wire [mean_N*bitwidth-1:0] x,
    input wire x_valid,
    input wire x_last,

    input wire [bitwidth-1:0] mean_variance,
    input wire [bitwidth-1:0] one_variance,

    output wire [mean_N*bitwidth-1:0] normal_out,
    output wire normal_out_valid,
    output wire normal_out_last,

    input wire [mean_N*bitwidth-1:0] weight,
    input wire [mean_N*bitwidth-1:0] bias,
    input wire param_valid,
    input wire param_last,

    output wire [mean_N*bitwidth-1:0] out,
    output wire out_valid,
    output wire out_last
);

wire [mean_N*bitwidth-1:0] mult_sub_out;
wire mult_sub_valid;
wire mult_sub_last;
wire [mean_N*bitwidth-1:0] mult_sub_in2;
assign mult_sub_in2 = mode ? 0 : mean_variance;
vector_mult_sub #(
    .bitwidth(bitwidth),
    .N(mean_N)
) u_vector_mult_sub (
    .clk(clk),
    .rstn(rstn),
    .in0(x),
    .in1(one_variance),
    .in2(mult_sub_in2),
    .in_valid(x_valid),
    .in_last(x_last),
    .out(mult_sub_out),
    .out_valid(mult_sub_valid),
    .out_last(mult_sub_last)
);

assign normal_out=mult_sub_out;
assign normal_out_valid=mult_sub_valid;
assign normal_out_last=mult_sub_last;

reg [mean_N*bitwidth-1:0] normal_out_q;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        normal_out_q <= 0;
    end
    else  begin
        normal_out_q <= normal_out;
    end
end

wire [mean_N*bitwidth-1:0] bias_mux;
assign bias_mux = mode ? 0 : bias;

vector_mult_add #(
    .bitwidth(bitwidth),
    .N(mean_N)
) u_vector_mult_add (
    .clk(clk),
    .rstn(rstn),
    .in0(normal_out_q),
    .in1(weight),
    .in2(bias_mux),
    .in_valid(param_valid),
    .in_last(param_last),
    .out(out),
    .out_valid(out_valid),
    .out_last(out_last)
);

endmodule