module BM_last_stage #(
    parameter bitwidth = 16,
    parameter N = 8,
    parameter hidden_num = 16
) (
    input wire clk,
    input wire rstn,

    input wire [bitwidth-1:0] gradb_add_gradg,
    input wire [bitwidth-1:0] gradb_sub_gradg,
    input wire [bitwidth-1:0] gradb,
    input wire [N*bitwidth-1:0] gradw_y2_mult,

    input wire mode, //0: hfln; 1: rms
    input wire [N*bitwidth-1:0] dz,
    input wire [N-1:0]          max_index,
    input wire [N-1:0]          min_index,
    input wire in_valid,
    input wire in_last,

    input wire [N*bitwidth-1:0] weight,
    input wire [bitwidth-1:0] variance_rec, ///m^(-1/2) for rms
    input wire param_valid,
    input wire param_last,

    output wire [N*bitwidth-1:0] gradx_out,
    output wire gradx_out_valid,
    output wire gradx_out_last
);

wire [N*bitwidth-1:0] floating_sub_out;
wire [N-1:0] floating_sub_valid;
wire [N-1:0] floating_sub_last;
genvar i;
generate
    for (i=0; i<N; i=i+1) begin
        
        reg [bitwidth-1:0] floating_sub_b;
        wire [1:0] index_choice;
        assign index_choice = {max_index[i], min_index[i]};
        always @(*) begin
            case (mode)
                1'b0: begin
                    case (index_choice)
                        2'b00:  floating_sub_b = gradb;
                        2'b01:  floating_sub_b = gradb_sub_gradg;
                        2'b10:  floating_sub_b = gradb_add_gradg;
                        default: floating_sub_b = gradb;
                    endcase
                end
                1'b1: begin
                    floating_sub_b = gradw_y2_mult[i*bitwidth +: bitwidth];
                end
                default: floating_sub_b = gradb;
            endcase
        end

        floating_sub u_floating_sub (
            .aclk(clk),
            .aresetn(rstn),
            .s_axis_a_tvalid(in_valid),
            .s_axis_a_tlast(in_last),
            .s_axis_a_tdata(dz[i*bitwidth +: bitwidth]),
            .s_axis_b_tvalid(in_valid),
            .s_axis_b_tlast(in_last),
            .s_axis_b_tdata(floating_sub_b),
            .m_axis_result_tvalid(floating_sub_valid[i]),
            .m_axis_result_tlast(floating_sub_last[i]),
            .m_axis_result_tdata(floating_sub_out[i*bitwidth +: bitwidth])
        );
    end
endgenerate

wire floating_sub_valid_w;
wire floating_sub_last_w;
assign floating_sub_valid_w = &floating_sub_valid;
assign floating_sub_last_w = &floating_sub_last;

reg [N*bitwidth-1:0] weight_delay [0:4];
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        weight_delay[0] <= 0;
        weight_delay[1] <= 0;
        weight_delay[2] <= 0;
        weight_delay[3] <= 0;
        weight_delay[4] <= 0;
    end
    else begin
        weight_delay[0] <= weight;
        weight_delay[1] <= weight_delay[0];
        weight_delay[2] <= weight_delay[1];
        weight_delay[3] <= weight_delay[2];
        weight_delay[4] <= weight_delay[3];
    end
end

reg [bitwidth-1:0] variance_rec_delay [0:4];
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        variance_rec_delay[0] <= 0;
        variance_rec_delay[1] <= 0;
        variance_rec_delay[2] <= 0;
        variance_rec_delay[3] <= 0;
        variance_rec_delay[4] <= 0;
    end
    else begin
        variance_rec_delay[0] <= variance_rec;
        variance_rec_delay[1] <= variance_rec_delay[0];
        variance_rec_delay[2] <= variance_rec_delay[1];
        variance_rec_delay[3] <= variance_rec_delay[2];
        variance_rec_delay[4] <= variance_rec_delay[3];
    end
end

reg [4:0] param_valid_delay;
reg [4:0] param_last_delay;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        param_valid_delay <= 0;
        param_last_delay <= 0;
    end
    else begin
        param_valid_delay <= {param_valid_delay[3:0], param_valid};  // 修正：应该是自己的移位
        param_last_delay <= {param_last_delay[3:0], param_last};
    end
end

wire [N*bitwidth-1:0] param_mult_out;
wire param_mult_valid;
wire param_mult_last;
vector_mult_vs #(
    .bitwidth   (bitwidth),
    .N          (N)
) u_vector_mult_vs_param(
    .clk(clk),
    .rstn(rstn),
    .in0(weight_delay[4]),
    .in1(variance_rec_delay[4]),
    .in_valid(param_valid_delay[4]),
    .in_last(param_last_delay[4]),
    .out(param_mult_out),
    .out_valid(param_mult_valid),
    .out_last(param_mult_last)
);

vector_mult #(
    .bitwidth   (bitwidth),
    .N          (N)
) u_vector_mult_2(
    .clk(clk),
    .rstn(rstn),
    .in0(param_mult_out),
    .in1(floating_sub_out),
    .in_valid(param_mult_valid & floating_sub_valid_w),
    .in_last(param_mult_last & floating_sub_last_w),
    .out(gradx_out),
    .out_valid(gradx_out_valid),
    .out_last(gradx_out_last)
);

    
endmodule