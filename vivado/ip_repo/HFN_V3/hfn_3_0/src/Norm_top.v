(* dont_touch = "yes" *)
module Norm_top #(
    parameter N = 8,
    parameter data_width = 128,
    parameter floating_width = 16,
    parameter burst_addr_width = 6,
    parameter sequence_length = 4,
    parameter hidden_dim = 512,
    parameter true_addr_width = $clog2(hidden_dim*sequence_length/N)
)(
    input wire clk,
    input wire resetn,

    input wire mode,  //0:HFLN 1:RMS

    //AXI FULL Port
    input  wire [burst_addr_width-1:0]   axi_full_addr,
    input  wire [data_width-1:0]   axi_full_din,
    output reg [data_width-1:0]   axi_full_dout,
    input  wire                    axi_full_we,
    
    // Control/Status signals from AXI-Lite
    input  wire [31:0]             ctrl_reg0,     //0:load_constant_en,1:load_constant_finish,2:FM_first_init_finish_x,3:FM_first_init_finish_w,4:FM_init_finish,5:FM_send_norm_finish,6:FM_send_out_finish,
                                                  //7:BM_param_inital_y_finish,8:BM_param_inital_dz_finish,9:BM_Param_send_result_finish           
                                                  //10:BM_dz_initial_finish,11:BM_send_result_finish
    input  wire [31:0]             ctrl_reg1,    // 0:load_constant_pulse,1:FM_en_pulse,2:BM_Param_en_pulse,3:BM_en_pulse

    input  wire [31:0]             constant0,    // H,y,dz
    input  wire [31:0]             constant1,    // C,y,dz(4)
    input  wire [31:0]             constant2,    // E,addr

    output wire  [31:0]             status_reg0,   //0:load_constant,1:FM_first_init_w,,2:FM_first_init_b,3:FM_init,4:FM_send_norm,5:FM_send_out,6:BM_param_idle,
                                                    //7:BM_param_inital_y,8:BM_param_inital_dz,9:BM_param_send_result
                                                    //10:BM_dz_initial,11:BM_send_result
    output reg  [31:0]             status_reg1,   // Status register 3
    output wire  [31:0]             bm_param_result,  //clk_cnt
    output wire  [31:0]             tst1,  
    output wire  [31:0]             tst2
);


parameter idle = 5'd0,
          load_constant = 5'd1,
          idle_FM = 5'd2,
          FM_first_initial_w = 5'd3,
          FM_first_initial_b = 5'd4,
          FM_work = 5'd5,
          FM_send_norm = 5'd6,
          FM_send_out = 5'd7,
          FM_initial = 5'd8,
          idle_BM_Param = 5'd9,
          BM_Param_initial_y = 5'd10,
          BM_Param_initial_dz = 5'd11,
          BM_Param_work = 5'd12,
          BM_Param_send_result = 5'd13,
          idle_BM = 5'd14,
          BM_initial_dz = 5'd15,
          BM_work = 5'd16,
          BM_send_result = 5'd17;

///////////////////////////*idle en*////////////////////////////////////////
assign load_constant_pulse= ctrl_reg1[0];
assign FM_en_pulse= ctrl_reg1[1];
wire BM_Param_en_pulse;
assign BM_Param_en_pulse = ctrl_reg1[2];  // FSM状态转换使能
wire BM_en_pulse;
assign BM_en_pulse = ctrl_reg1[3];
///////////////////////////*clk*////////////////////////////////////////
reg [31:0] FM_cal_clk;
reg [31:0] BM_Param_cal_clk;
reg [31:0] BM_cal_clk;
always @(*) begin
    case ({mode_en[17],mode_en[13],mode_en[7]})
        3'b001: status_reg1 = FM_cal_clk;
        3'b010: status_reg1 = BM_Param_cal_clk;
        3'b100: status_reg1 = BM_cal_clk;
        default: status_reg1 = 0;
    endcase
end

// tst1: 锁存BM调试信号 - gradb_add_gradg和gradb
// tst2: 锁存BM调试信号 - gradb_sub_gradg和variance_rec
reg [floating_width-1:0] gradb_add_gradg_locked;
reg [floating_width-1:0] gradb_locked;
reg [floating_width-1:0] gradb_sub_gradg_locked;
reg [floating_width-1:0] variance_rec_locked;

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        gradb_add_gradg_locked <= 0;
        gradb_locked <= 0;
        gradb_sub_gradg_locked <= 0;
        variance_rec_locked <= 0;
    end
    else if (mode_en[16] && BM_top_gradx_out_valid) begin
        // 在BM工作期间有输出时锁存第一阶段的计算结果
        if (BM_top_second_stage_addr == 0) begin
            // 锁存地址0时的第一阶段计算结果用于调试
            gradb_add_gradg_locked <= BM_top_gradb_add_gradg_tst;
            gradb_locked <= BM_top_gradb_tst;
            gradb_sub_gradg_locked <= BM_top_gradb_sub_gradg_tst;
            variance_rec_locked <= variance_rec;
        end
    end
end

assign tst1 = {gradb_add_gradg_locked, gradb_locked};        // 32位: {gradb_add_gradg[15:0], gradb[15:0]}
assign tst2 = {gradb_sub_gradg_locked, variance_rec_locked}; // 32位: {gradb_sub_gradg[15:0], variance_rec[15:0]}

//FSM
reg [4:0] state,next_state;
always @(posedge clk or negedge resetn) begin
    if(!resetn)
        state<=idle;
    else
        state<=next_state;
end

//load constant signals
wire load_constant_pulse;
wire load_constant_finish;

//FM signals
wire       FM_en_pulse;
wire       FM_first_initial_finish;
wire       FM_send_norm_finish;
wire       FM_send_out_finish;
wire       BM_Param_inital_y_finish;
wire       BM_Param_inital_dz_finish;
wire       BM_Param_send_result_finish;
wire       BM_dz_initial_finish;
wire       BM_send_result_finish;

always @(*) begin
    case (state)
        idle:begin
            if (load_constant_pulse) begin
                next_state=load_constant;
            end
            else begin
                next_state=idle;
            end
        end
        load_constant:begin
            if (load_constant_finish) begin
                next_state=idle_FM;
            end
            else begin
                next_state=load_constant;
            end
        end
        idle_FM:begin
            if (FM_en_pulse) begin
                next_state=FM_first_initial_w;
            end
            else begin
                next_state=idle_FM;
            end
        end
        FM_first_initial_w:begin
            if (FM_first_initial_finish_w) begin
                next_state=FM_first_initial_b;
            end
            else begin
                next_state=FM_first_initial_w;
            end
        end
        FM_first_initial_b:begin
            if (FM_first_initial_finish_b) begin
                next_state=FM_initial;
            end
            else begin
                next_state=FM_first_initial_b;
            end
        end
        FM_initial:begin
            if (FM_initial_finish) begin
                next_state=FM_work;
            end
            else begin
                next_state=FM_initial;
            end 
        end
        FM_work:begin
            if (fm_work_finish) begin
                next_state=FM_send_norm;
            end
            else begin
                next_state=FM_work;
            end
        end
        FM_send_norm:begin
            if(FM_send_norm_finish)begin
                next_state=FM_send_out;
            end
            else begin
                next_state = FM_send_norm;
            end
        end
        FM_send_out:begin
            if(FM_send_out_finish)begin
                next_state=idle_BM_Param;
            end
            else begin
                next_state = FM_send_out;
            end
        end
        idle_BM_Param:begin
            if (BM_Param_en_pulse) begin
                next_state = BM_Param_initial_y;
            end
            else begin
                next_state = idle_BM_Param;
            end
        end
        BM_Param_initial_y:begin
            if (BM_Param_inital_y_finish) begin
                next_state = BM_Param_initial_dz;
            end
            else begin
                next_state = BM_Param_initial_y;
            end
        end
        BM_Param_initial_dz:begin
            if (BM_Param_inital_dz_finish) begin
                next_state = BM_Param_work;
            end
            else begin
                next_state = BM_Param_initial_dz;
            end
        end
        BM_Param_work:begin
            if (BM_Param_work_finish) begin
                next_state = BM_Param_send_result;
            end
            else begin
                next_state = BM_Param_work;
            end
        end
        BM_Param_send_result:begin
            if (BM_Param_send_result_finish) begin
                next_state = idle_BM;
            end
            else begin
                next_state = BM_Param_send_result;
            end
        end
        idle_BM:begin
            if (BM_en_pulse) begin
                next_state = BM_initial_dz;
            end
            else begin
                next_state = idle_BM;
            end
        end
        BM_initial_dz:begin
            if (BM_dz_initial_finish) begin
                next_state = BM_work;
            end
            else begin
                next_state = BM_initial_dz;
            end
        end
        BM_work:begin
            if (BM_work_finish) begin
                next_state = BM_send_result;
            end
            else begin
                next_state = BM_work;
            end
        end
        BM_send_result:begin
            if (BM_send_result_finish) begin
                next_state = idle;
            end
            else begin
                next_state = BM_send_result;
            end
        end
        default:begin
            next_state=idle;
        end
    endcase
end

reg [21:0] mode_en;
always @(posedge clk or negedge resetn) begin
    if(!resetn)begin
      mode_en<=0;
    end
    else begin
        case (state)
            idle:begin
                mode_en<=22'b00_0000_0000_0000_0000_0001;
            end
            load_constant:begin
                mode_en<=22'b00_0000_0000_0000_0000_0010;
            end
            idle_FM:begin
                mode_en<=22'b00_0000_0000_0000_0000_0100;
            end
            FM_first_initial_w:begin
                mode_en<=22'b00_0000_0000_0000_0000_1000;
            end
            FM_first_initial_b:begin
                mode_en<=22'b00_0000_0000_0000_0001_0000;
            end
            FM_initial:begin
                mode_en<=22'b00_0000_0000_0000_0010_0000;
            end
            FM_work:begin
                mode_en<=22'b00_0000_0000_0000_0100_0000;
            end
            FM_send_norm:begin
                mode_en<=22'b00_0000_0000_0000_1000_0000;
            end
            FM_send_out:begin
                mode_en<=22'b00_0000_0000_0001_0000_0000;
            end
            idle_BM_Param:begin
                mode_en<=22'b00_0000_0000_0010_0000_0000;
            end
            BM_Param_initial_y:begin
                mode_en<=22'b00_0000_0000_0100_0000_0000;
            end
            BM_Param_initial_dz:begin
                mode_en<=22'b00_0000_0000_1000_0000_0000;
            end
            BM_Param_work:begin
                mode_en<=22'b00_0000_0001_0000_0000_0000;
            end
            BM_Param_send_result:begin
                mode_en<=22'b00_0000_0010_0000_0000_0000;
            end
            idle_BM:begin
                mode_en<=22'b00_0000_0100_0000_0000_0000;
            end
            BM_initial_dz:begin
                mode_en<=22'b00_0000_1000_0000_0000_0000;
            end
            BM_work:begin
                mode_en<=22'b00_0001_0000_0000_0000_0000;
            end
            BM_send_result:begin
                mode_en<=22'b00_0010_0000_0000_0000_0000;
            end
            default: begin
                mode_en<=22'b00_0000_0000_0000_0000_0001;
            end
        endcase
    end
end


/////////////////////////////////*load constant to reg*//////////////////////////////////
assign status_reg0[0]=mode_en[1];
assign load_constant_finish=ctrl_reg0[1];
reg [floating_width-1:0]  constant_value   [0:2];   //0:reciprocal_last_dim,1:variance_divider,2:eps
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        constant_value[0]<=0;
        constant_value[1]<=0;
        constant_value[2]<=0;
    end
    else if (mode_en[1]) begin
        constant_value[0]<=constant0[floating_width-1:0];
        constant_value[1]<=constant1[floating_width-1:0];
        constant_value[2]<=constant2[floating_width-1:0];
    end
    else begin
        constant_value[0]<=constant_value[0];
        constant_value[1]<=constant_value[1];
        constant_value[2]<=constant_value[2];
    end
end

///////////////////////////*FM first W init*////////////////////////////////////////
assign status_reg0[1]= mode_en[3];
assign FM_first_initial_finish_w=ctrl_reg0[2];
///////////////////////////*FM first B init*////////////////////////////////////////
assign status_reg0[2]=mode_en[4];
assign FM_first_initial_finish_b=ctrl_reg0[3];
///////////////////////*FM init*///////////////////////////////////////////////////
assign status_reg0[3]=mode_en[5];
assign FM_initial_finish=ctrl_reg0[4];
/////////////////////////*FM send norm*//////////////////////////////////////////
assign status_reg0[4]=mode_en[7];
assign FM_send_norm_finish=ctrl_reg0[5];
/////////////////////////*FM send out*//////////////////////////////////////////
assign status_reg0[5]=mode_en[8];
assign FM_send_out_finish=ctrl_reg0[6];
////////////////////////*BM Param idle*////////////////////////////////////////
assign status_reg0[6]=mode_en[9];
////////////////////////*BM Param y init*////////////////////////////////////////
assign status_reg0[7]=mode_en[10];
assign BM_Param_inital_y_finish=ctrl_reg0[7];
////////////////////////*BM Param dz init*////////////////////////////////////////
assign status_reg0[8]=mode_en[11];
assign BM_Param_inital_dz_finish=ctrl_reg0[8];
////////////////////////*BM Param send result*////////////////////////////////////////
assign status_reg0[9]=mode_en[13];
assign BM_Param_send_result_finish=ctrl_reg0[9];
////////////////////////*BM initial dz*////////////////////////////////////////
assign status_reg0[10]=mode_en[15];
assign BM_dz_initial_finish=ctrl_reg0[10];
////////////////////////*BM send result*////////////////////////////////////////
assign status_reg0[11]=mode_en[17];
assign BM_send_result_finish=ctrl_reg0[11];

always @(*) begin
    case ({mode_en[17],mode_en[15],mode_en[8],mode_en[7],mode_en[5],mode_en[4],mode_en[3]})
        7'b0000001: axi_full_dout = bram_w_doutb;      // FM_first_initial_w (读W)
        7'b0000010: axi_full_dout = bram_b_doutb;      // FM_first_initial_b (读B)
        7'b0000100: axi_full_dout = bram_x_doutb;      // FM_initial (读X)
        7'b0001000: axi_full_dout = bram_norm_doutb;   // FM_send_norm (读Norm)
        7'b0010000: axi_full_dout = bram_fm_doutb;     // FM_send_out (读FM输出)
        7'b0100000: axi_full_dout = bram_dz_doutb;     // BM_initial_dz (读DZ) ← 添加！
        7'b1000000: axi_full_dout = bram_bm_doutb;     // BM_send_result (读Gradx) ← 关键修复！
        default: axi_full_dout = 0;
    endcase
end


//BRAM W
wire [true_addr_width-1:0] bram_w_addra;
reg [true_addr_width-1:0] bram_w_addrb;
wire [data_width-1:0] bram_w_dina;
wire [data_width-1:0] bram_w_doutb;
wire bram_w_wea;

wire [data_width-1:0] FM_top_weight_data;
wire [true_addr_width-1:0] FM_top_param_addr;
assign FM_top_weight_data = mode_en[6] ? bram_w_doutb : 0;
assign BM_top_w_data = mode_en[16] ? bram_w_doutb : 0;

assign bram_w_addra = mode_en[3] ? {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr} : 0;
assign bram_w_dina = mode_en[3] ? axi_full_din : 0;
assign bram_w_wea = mode_en[3] ? axi_full_we : 0;
(* dont_touch = "yes" *) bram_w  u_bram_w(
    // Port A - AXI-Full side
    .addra  (bram_w_addra),
    .clka   (clk),
    .dina   (bram_w_dina),
    .douta  (),
    .wea    (bram_w_wea),
    
    // Port B - Computation side
    .addrb  (bram_w_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_w_doutb),
    .web    (0)
);

always @(*) begin
    case ({mode_en[16],mode_en[6],mode_en[3]})
        3'b001: bram_w_addrb = {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr};  // 高位�??0，完整赋�??
        3'b010: bram_w_addrb = FM_top_param_addr;
        3'b100: bram_w_addrb = BM_top_second_stage_addr;
        default:bram_w_addrb = 0;
    endcase
end


//BRAM B
wire [true_addr_width-1:0] bram_b_addra;
reg [true_addr_width-1:0] bram_b_addrb;
wire [data_width-1:0] bram_b_dina;
wire [data_width-1:0] bram_b_doutb;
wire bram_b_wea;

wire [data_width-1:0] FM_top_bias_data;
assign FM_top_bias_data = mode_en[6] ? bram_b_doutb : 0;


assign bram_b_addra = mode_en[4] ? {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr} : 0;
assign bram_b_dina = mode_en[4] ? axi_full_din : 0;
assign bram_b_wea = mode_en[4] ? axi_full_we : 0;
(* dont_touch = "yes" *) bram_b  u_bram_b(
    // Port A - AXI-Full side
    .addra  (bram_b_addra),
    .clka   (clk),
    .dina   (bram_b_dina),
    .douta  (),
    .wea    (bram_b_wea),
    
    // Port B - Computation side
    .addrb  (bram_b_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_b_doutb),
    .web    (0)
);

always @(*) begin
    case ({mode_en[6],mode_en[4]})
        2'b01: bram_b_addrb = {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr};
        2'b10: bram_b_addrb = FM_top_param_addr;
        default: bram_b_addrb = 0;
    endcase
end

//BRAM X
reg [true_addr_width-1:0] bram_x_addra;
reg [true_addr_width-1:0] bram_x_addrb;
wire [data_width-1:0] bram_x_dina;
wire [data_width-1:0] bram_x_douta;
wire [data_width-1:0] bram_x_doutb;
wire bram_x_wea;

wire [true_addr_width-1:0] FM_top_x1_addr;
wire [data_width-1:0] FM_top_x1_data;
wire [true_addr_width-1:0] FM_top_x2_addr;
wire [data_width-1:0] FM_top_x2_data;
assign FM_top_x1_data = mode_en[6] ? bram_x_douta : 0;
assign FM_top_x2_data = mode_en[6] ? bram_x_doutb : 0;


always @(*) begin
    case ({mode_en[6],mode_en[5]})
        2'b01: bram_x_addra = {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr};
        2'b10: bram_x_addra = FM_top_x1_addr;
        default: bram_x_addra = 0;
    endcase
end

assign bram_x_dina = mode_en[5] ? axi_full_din : 0;
assign bram_x_wea = mode_en[5] ? axi_full_we : 0;
(* dont_touch = "yes" *) bram_x  u_bram_x(
    // Port A - AXI-Full side
    .addra  (bram_x_addra),
    .clka   (clk),
    .dina   (bram_x_dina),
    .douta  (bram_x_douta),
    .wea    (bram_x_wea),
    
    // Port B - Computation side
    .addrb  (bram_x_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_x_doutb),
    .web    (0)
);

always @(*) begin
    case ({mode_en[6],mode_en[5]})
        2'b01: bram_x_addrb = {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr};
        2'b10: bram_x_addrb = FM_top_x2_addr;
        default: bram_x_addrb = 0;
    endcase
end

//FM_top
wire FM_cal_en_pulse;
reg FM_cal_en_q1;
assign FM_cal_en_pulse=(mode_en[6] & ~FM_cal_en_q1);
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        FM_cal_en_q1<=0;
    end
    else begin
        FM_cal_en_q1<=mode_en[6];
    end
end

(* dont_touch = "yes" *) FM_top #(
    .hidden_states(hidden_dim),
    .sequence_length(sequence_length),
    .bitwidth(floating_width),
    .std_N(N),
    .mean_N(N)
)u_FM_top(
    .clk(clk),
    .rstn(resetn),
    .en(FM_cal_en_pulse),

    .mode(mode),

    .x1(FM_top_x1_data),
    .x1_addr(FM_top_x1_addr),

    .x2(FM_top_x2_data),
    .x2_addr(FM_top_x2_addr),

    .weight(FM_top_weight_data),
    .bias(FM_top_bias_data),
    .param_addr(FM_top_param_addr),

    .H_param(constant_value[0]),
    .C_param(constant_value[1]),
    .E_param(constant_value[2]),

    .normal_out(FM_top_normal_data),
    .normal_out_valid(FM_top_normal_valid),
    .normal_out_last(FM_top_normal_last),

    .one_variance(one_variance),
    .FM_out(FM_top_fm_data),
    .FM_out_valid(FM_top_fm_valid),
    .FM_out_last(FM_top_fm_last),

    .max_index(fm_max_index),
    .min_index(fm_min_index),
    .index_valid(fm_index_valid),
    .index_last(fm_index_last),

    .mean_out_r_t(mean_out_r_t),
    .variance_out_r_t(variance_out_r_t),
    .mean_variance_out_r_t(mean_variance_out_r_t),
    .one_variance_out_r_t(one_variance_out_r_t)
);
wire [floating_width-1:0] one_variance;
wire [floating_width-1:0] mean_out_r_t;
wire [floating_width-1:0] variance_out_r_t;
wire [floating_width-1:0] mean_variance_out_r_t;
wire [floating_width-1:0] one_variance_out_r_t;


reg FM_cal_clk_en;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        FM_cal_clk_en <= 0;
    end
    else if (FM_cal_en_pulse) begin
        FM_cal_clk_en <= 1;
    end
    else if (fm_work_finish) begin
        FM_cal_clk_en <= 0;
    end
end

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        FM_cal_clk <= 0;
    end
    else if (FM_cal_en_pulse) begin
        FM_cal_clk <= 0;
    end
    else if (FM_cal_clk_en && !fm_work_finish) begin
        FM_cal_clk <= FM_cal_clk + 1;
    end
end


//BRAM norm
reg [true_addr_width-1:0] bram_norm_addra;
reg [true_addr_width-1:0] bram_norm_addra_cnt;
wire [data_width-1:0] bram_norm_doutb;
wire [data_width-1:0] bram_norm_douta;
reg [true_addr_width-1:0] bram_norm_addrb;

wire [data_width-1:0] FM_top_normal_data;
wire FM_top_normal_valid;
wire FM_top_normal_last;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_norm_addra_cnt <= 0;
    end
    else if (FM_top_normal_valid & ~FM_top_normal_last) begin
        bram_norm_addra_cnt <= bram_norm_addra_cnt + 1;
    end
    else if (FM_top_normal_valid & FM_top_normal_last) begin
        bram_norm_addra_cnt <= 0;
    end
end


assign bram_norm_outa_en = mode & mode_en[16];
always @(*) begin
    case ({bram_norm_outa_en,mode_en[6]})
        2'b01: bram_norm_addra = bram_norm_addra_cnt + bram_norm_addrb_base;
        2'b10: bram_norm_addra = BM_top_norm2_addr;
        default: bram_norm_addra = 0;
    endcase
end


localparam block_size = hidden_dim/N;
assign last_seq_norm = (bram_norm_addrb_base==(sequence_length-1)*block_size);
reg [true_addr_width-1:0] bram_norm_addrb_base;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_norm_addrb_base <= 0;
    end
    else if (FM_top_normal_valid & FM_top_normal_last) begin
        if (last_seq_norm) begin
            bram_norm_addrb_base <= 0;
        end
        else begin
            bram_norm_addrb_base <= bram_norm_addrb_base + block_size;
        end
    end
end

always @(*) begin
    case ({mode_en[16],mode_en[7]})
        2'b01: bram_norm_addrb = {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr};
        2'b10: bram_norm_addrb = BM_top_dz_y_addr;
        default: bram_norm_addrb = 0;
    endcase
end

// assign bram_norm_addrb = mode_en[7] ? {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr} : 0;
(* dont_touch = "yes" *) bram_norm  u_bram_norm(
    .addra  (bram_norm_addra),
    .clka   (clk),
    .dina   (FM_top_normal_data),
    .douta  (bram_norm_douta),
    .wea    (FM_top_normal_valid),

    .addrb  (bram_norm_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_norm_doutb),
    .web    (0)
);

assign BM_top_norm2_data = (mode_en[16] & mode) ? bram_norm_douta : 0;
assign BM_top_norm_data = mode_en[16] ? bram_norm_doutb : 0;

//BRAM fm_out
wire fm_work_finish;
assign fm_work_finish = FM_top_fm_valid & FM_top_fm_last & last_seq_fm;

reg [true_addr_width-1:0] bram_fm_addra_cnt;
wire [true_addr_width-1:0] bram_fm_addra;
wire [data_width-1:0] bram_fm_doutb;
wire [true_addr_width-1:0] bram_fm_addrb;

wire [data_width-1:0] FM_top_fm_data;
wire  FM_top_fm_valid;
wire  FM_top_fm_last;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_fm_addra_cnt <= 0;
    end
    else if (FM_top_fm_valid & ~FM_top_fm_last) begin
        bram_fm_addra_cnt <= bram_fm_addra_cnt + 1;
    end
    else if (FM_top_fm_valid & FM_top_fm_last) begin
        bram_fm_addra_cnt <= 0;
    end
end
assign bram_fm_addra = bram_fm_addra_cnt + bram_fm_addra_base;

assign last_seq_fm = (bram_fm_addra_base==(sequence_length-1)*block_size);
reg [true_addr_width-1:0] bram_fm_addra_base;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_fm_addra_base <= 0;
    end
    else if (FM_top_fm_valid & FM_top_fm_last) begin
        if (last_seq_fm) begin
            bram_fm_addra_base <= 0;
        end
        else begin
            bram_fm_addra_base <= bram_fm_addra_base + block_size;
        end
    end
end


assign bram_fm_addrb = mode_en[8] ? {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr} : 0;
(* dont_touch = "yes" *) bram_fm  u_bram_fm(
    .addra  (bram_fm_addra),
    .clka   (clk),
    .dina   (FM_top_fm_data),
    .douta  (),
    .wea    (FM_top_fm_valid),

    .addrb  (bram_fm_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_fm_doutb),
    .web    (0)
);

//Reg max min index
reg [N-1:0] reg_max_index [0:(hidden_dim*sequence_length)/N-1];
reg [N-1:0] reg_min_index [0:(hidden_dim*sequence_length)/N-1];

wire [N-1:0] fm_max_index;
wire [N-1:0] fm_min_index;
wire fm_index_valid;
wire fm_index_last;

reg [$clog2(hidden_dim/N)-1:0] reg_index_addr;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        reg_index_addr <= 0;
    end
    else if (fm_index_valid & ~fm_index_last) begin
        reg_index_addr <= reg_index_addr + 1;
    end
    else if (fm_index_valid & fm_index_last) begin
        reg_index_addr <= 0;
    end
end

integer i;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        for (i=0; i<(hidden_dim/N); i=i+1) begin
            reg_max_index[i] <= 0;
            reg_min_index[i] <= 0;
        end
    end
    else if(fm_index_valid) begin
        reg_max_index[reg_index_addr+reg_index_addr_base] <= fm_max_index;
        reg_min_index[reg_index_addr+reg_index_addr_base] <= fm_min_index;
    end
end

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        BM_top_max_index <= 0;
        BM_top_min_index <= 0;
    end
    else if(mode_en[16]) begin
        BM_top_max_index <= reg_max_index[BM_top_second_stage_addr];
        BM_top_min_index <= reg_min_index[BM_top_second_stage_addr];
    end
    else begin
        BM_top_max_index <= 0;
        BM_top_min_index <= 0;
    end
end


assign last_seq_index = (reg_index_addr_base==(sequence_length-1)*block_size);
reg [true_addr_width-1:0] reg_index_addr_base;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        reg_index_addr_base <= 0;
    end
    else if (fm_index_valid & fm_index_last) begin
        if (last_seq_index) begin
            reg_index_addr_base <= 0;
        end
        else begin
            reg_index_addr_base <= reg_index_addr_base + block_size;
        end
    end
end


integer j;
reg [floating_width-1:0] one_variance_r [0:sequence_length-1];
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        for (j=0;j<sequence_length;j=j+1) begin
            one_variance_r[j] <= 0;
        end
    end
    else if (FM_top_normal_last) begin
        one_variance_r[variance_rec_fm_addr] <= one_variance;
    end
end

reg [$clog2(sequence_length)-1:0] variance_rec_fm_addr;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        variance_rec_fm_addr <= 0;
    end
    else if (FM_top_normal_last) begin
        variance_rec_fm_addr <= variance_rec_fm_addr + 1;
    end
    else if (fm_work_finish) begin
        variance_rec_fm_addr <= 0;
    end
end


always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        variance_rec <= 0;
    end
    else if (mode_en[16]) begin
        variance_rec <= one_variance_r[variance_rec_addr];
    end
end

///////////////////////////////BM Param/////////////////////////////////////////////////////
wire BM_Param_work_finish;

localparam  BM_Param_addr_bitwidth = $clog2(hidden_dim);
localparam  BM_Param_data_width = floating_width*sequence_length;

//BRAM param y
wire [BM_Param_addr_bitwidth-1:0] bram_param_y_addra;
wire [BM_Param_addr_bitwidth-1:0] bram_param_y_addrb;
wire [BM_Param_data_width-1:0] bram_param_y_dina;
wire [BM_Param_data_width-1:0] bram_param_y_doutb;
wire bram_param_y_wea;

wire [BM_Param_data_width-1:0] BM_Param_y_data;
wire [BM_Param_addr_bitwidth-1:0] BM_Param_in_addr;
assign BM_Param_y_data = mode_en[12] ? bram_param_y_doutb : 0;
assign bram_param_y_addrb = mode_en[12] ? BM_Param_in_addr : 0;

assign bram_param_y_addra = mode_en[10] ? constant2[BM_Param_addr_bitwidth-1:0] : 0;
assign bram_param_y_dina = mode_en[10] ? {constant0,constant1} : 0;
assign bram_param_y_wea = mode_en[10];
(* dont_touch = "yes" *) bram_param_y  u_bram_param_y(
    .addra  (bram_param_y_addra),
    .clka   (clk),
    .dina   (bram_param_y_dina),
    .douta  (),
    .wea    (bram_param_y_wea),
    
    .addrb  (bram_param_y_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_param_y_doutb),
    .web    (0)
);

//BRAM param dz
wire [BM_Param_addr_bitwidth-1:0] bram_param_dz_addra;
wire [BM_Param_addr_bitwidth-1:0] bram_param_dz_addrb;
wire [BM_Param_data_width-1:0] bram_param_dz_dina;
wire [BM_Param_data_width-1:0] bram_param_dz_doutb;
wire bram_param_dz_wea;

wire [BM_Param_data_width-1:0] BM_Param_dz_data;
assign BM_Param_dz_data = mode_en[12] ? bram_param_dz_doutb : 0;
assign bram_param_dz_addrb = mode_en[12] ? BM_Param_in_addr : 0;

assign bram_param_dz_addra = mode_en[11] ? constant2[BM_Param_addr_bitwidth-1:0] : 0;
assign bram_param_dz_dina = mode_en[11] ? {constant0,constant1} : 0;
assign bram_param_dz_wea = mode_en[11];
(* dont_touch = "yes" *) bram_param_dz  u_bram_param_dz(
    .addra  (bram_param_dz_addra),
    .clka   (clk),
    .dina   (bram_param_dz_dina),
    .douta  (),
    .wea    (bram_param_dz_wea),
    
    .addrb  (bram_param_dz_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_param_dz_doutb),
    .web    (0)
);


wire [floating_width-1:0] gradg_out;
wire [floating_width-1:0] gradb_out;
wire gradg_out_valid;
wire gradg_out_last;
wire gradb_out_valid;
wire gradb_out_last;
wire BM_Param_work_en_pulse;
reg BM_Param_en_q1;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        BM_Param_en_q1 <= 0;
    end
    else begin
        BM_Param_en_q1 <= mode_en[12];
    end
end
assign BM_Param_work_en_pulse = mode_en[12] & ~BM_Param_en_q1;
(* dont_touch = "yes" *) BM_param_top #(
    .bitwidth(floating_width),
    .sequence_length(sequence_length),
    .hidden_dim(hidden_dim)
) u_BM_param_top (
    .clk(clk),
    .rstn(resetn),

    .en(BM_Param_work_en_pulse),

    .gradz(BM_Param_dz_data),
    .y1(BM_Param_y_data),
    .addr(BM_Param_in_addr),

    .gradg_out(gradg_out),
    .gradg_out_valid(gradg_out_valid),
    .gradg_out_last(gradg_out_last),

    .gradb_out(gradb_out),
    .gradb_out_valid(gradb_out_valid),
    .gradb_out_last(gradb_out_last)
);

assign BM_Param_work_finish = gradg_out_last;

reg BM_Param_cal_clk_en;
always @(posedge clk or negedge resetn) begin
   if (!resetn) begin
      BM_Param_cal_clk_en <= 0;
   end 
   else if (BM_Param_work_en_pulse) begin
        BM_Param_cal_clk_en <= 1;
    end
    else if (BM_Param_work_finish) begin
        BM_Param_cal_clk_en <= 0;
    end
end

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        BM_Param_cal_clk <= 0;
    end
    else if (BM_Param_work_en_pulse) begin
        BM_Param_cal_clk <= 0;
    end
    else if (BM_Param_cal_clk_en && !BM_Param_work_finish) begin
        BM_Param_cal_clk <= BM_Param_cal_clk + 1;
    end
end


//BRAM param gradg
reg [BM_Param_addr_bitwidth-1:0] bram_param_gradg_addra;
wire [BM_Param_addr_bitwidth-1:0] bram_param_gradg_addrb;
wire [floating_width-1:0] bram_param_gradg_doutb;

assign bram_param_gradg_addrb = mode_en[13] ? constant2[BM_Param_addr_bitwidth-1:0]  : 0;

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_param_gradg_addra <= 0;
    end
    else if (gradg_out_valid & ~gradg_out_last) begin
        bram_param_gradg_addra <= bram_param_gradg_addra + 1;
    end
    else if 
        (gradg_out_valid & gradg_out_last) begin
        bram_param_gradg_addra <= 0;
    end
end

(* dont_touch = "yes" *) bram_param_gradg  u_bram_param_gradg (
    .addra  (bram_param_gradg_addra),
    .clka   (clk),
    .dina   (gradg_out),
    .douta  (),
    .wea    (gradg_out_valid),
    
    .addrb  (bram_param_gradg_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_param_gradg_doutb),
    .web    (0)
);


//BRAM param gradb
reg [BM_Param_addr_bitwidth-1:0] bram_param_gradb_addra;
wire [BM_Param_addr_bitwidth-1:0] bram_param_gradb_addrb;
wire [floating_width-1:0] bram_param_gradb_doutb;

assign bram_param_gradb_addrb = mode_en[13] ? constant2[BM_Param_addr_bitwidth-1:0]  : 0;

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_param_gradb_addra <= 0;
    end
    else if (gradb_out_valid & ~gradb_out_last) begin
        bram_param_gradb_addra <= bram_param_gradb_addra + 1;
    end
    else if 
        (gradb_out_valid & gradb_out_last) begin
        bram_param_gradb_addra <= 0;
    end
end

(* dont_touch = "yes" *) bram_param_gradb  u_bram_param_gradb(
    .addra  (bram_param_gradb_addra),
    .clka   (clk),
    .dina   (gradb_out),
    .douta  (),
    .wea    (gradb_out_valid),
    
    .addrb  (bram_param_gradb_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_param_gradb_doutb),
    .web    (0)
);

assign bm_param_result = mode_en[13] ? {bram_param_gradg_doutb,bram_param_gradb_doutb} : 0;

///////////////////////////////BM Work/////////////////////////////////////////////////////

//BRAM dz
reg [true_addr_width-1:0] bram_dz_addra;
reg [true_addr_width-1:0] bram_dz_addrb;
wire [data_width-1:0] bram_dz_dina;
wire [data_width-1:0] bram_dz_douta;
wire [data_width-1:0] bram_dz_doutb;
wire bram_dz_wea;

wire [data_width-1:0] BM_top_dz1_data;
wire [data_width-1:0] BM_top_dz2_data;
assign BM_top_dz1_data = mode_en[16] ? bram_dz_douta : 0;
assign BM_top_dz2_data = mode_en[16] ? bram_dz_doutb : 0;


always @(*) begin
    case ({mode_en[16],mode_en[15]})
        2'b01: bram_dz_addra = {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr};
        2'b10: bram_dz_addra = BM_top_dz_y_addr;  // 使用BM_top输出的dz_y_addr，用于第一阶段读取dz1
        default: bram_dz_addra = 0;
    endcase
end

assign bram_dz_dina = mode_en[15] ? axi_full_din : 0;
assign bram_dz_wea = mode_en[15] ? axi_full_we : 0;
(* dont_touch = "yes" *) bram_dz  u_bram_dz(
    .addra  (bram_dz_addra),
    .clka   (clk),
    .dina   (bram_dz_dina),
    .douta  (bram_dz_douta),
    .wea    (bram_dz_wea),
    
    .addrb  (bram_dz_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_dz_doutb),
    .web    (0)
);

always @(*) begin
    case ({mode_en[16],mode_en[15]})
        2'b01: bram_dz_addrb = {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr};
        2'b10: bram_dz_addrb = BM_top_second_stage_addr;
        default: bram_dz_addrb = 0;
    endcase
end

wire [data_width-1:0] BM_top_norm_data;
wire [true_addr_width-1:0] BM_top_dz_y_addr;
reg [N-1:0] BM_top_max_index;
reg [N-1:0] BM_top_min_index;
wire [data_width-1:0] BM_top_w_data;
wire [true_addr_width-1:0] BM_top_second_stage_addr;
wire [data_width-1:0] BM_top_gradx_out;
wire BM_top_gradx_out_valid;
wire BM_top_gradx_out_last;
wire [floating_width-1:0] BM_top_gradb_add_gradg_tst;
wire [floating_width-1:0] BM_top_gradb_tst;
wire [floating_width-1:0] BM_top_gradb_sub_gradg_tst;
reg [floating_width-1:0] variance_rec;

//RMS Ports
wire [true_addr_width-1:0] BM_top_norm2_addr;
wire [data_width-1:0] BM_top_norm2_data;

wire BM_work_en_pulse;
reg BM_en_pulse_d1;
assign BM_work_en_pulse = mode_en[16] & ~BM_en_pulse_d1;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        BM_en_pulse_d1 <= 0;
    end
    else begin
        BM_en_pulse_d1 <= mode_en[16];
    end
end


(* dont_touch = "yes" *)  BM_top #(
    .hidden_p(hidden_dim),
    .sequence_length(sequence_length),
    .bitwidth(floating_width),
    .N(N)
)u_BM_top(
    .clk            (clk),
    .rstn           (resetn),
    .en             (BM_work_en_pulse),

    .mode           (mode),

    .dz1            (BM_top_dz1_data),
    .y             (BM_top_norm_data),
    .dz_y_addr      (BM_top_dz_y_addr),

    .y2             (BM_top_norm2_data),
    .y2_addr        (BM_top_norm2_addr),
    
    .dz2            (BM_top_dz2_data),
    .max_index      (BM_top_max_index),
    .min_index      (BM_top_min_index),
    .weight         (BM_top_w_data),
    .second_stage_addr(BM_top_second_stage_addr),

    .H_param        (constant_value[0]),  // 使用16位的constant_value，与FM_top保持一致
    .C_param        (constant_value[1]),  // 使用16位的constant_value，与FM_top保持一致
    .variance_rec   (variance_rec),

    .gradx_out      (BM_top_gradx_out),
    .gradx_out_valid(BM_top_gradx_out_valid),
    .gradx_out_last (BM_top_gradx_out_last)
);


reg BM_cal_clk_en;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        BM_cal_clk_en <= 0;
    end
    else if (BM_work_en_pulse) begin
        BM_cal_clk_en <= 1;
    end
    else if (BM_work_finish) begin
        BM_cal_clk_en <= 0;
    end
end

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        BM_cal_clk <= 0;
    end
    else if (BM_work_en_pulse) begin
        BM_cal_clk <= 0;
    end
    else if (BM_cal_clk_en && !BM_work_finish) begin
        BM_cal_clk <= BM_cal_clk + 1;
    end
end


//BRAM gradx (BM result output)
// 先定义地址基址寄存器
reg [true_addr_width-1:0] bram_bm_addra_base;
// 基于基址定义last_seq标志
wire last_seq_bm;
assign last_seq_bm = (bram_bm_addra_base==(sequence_length-1)*block_size);
// 定义完成标志
wire BM_work_finish;
assign BM_work_finish = BM_top_gradx_out_valid & BM_top_gradx_out_last & last_seq_bm;

reg [true_addr_width-1:0] bram_bm_addra_cnt;
wire [true_addr_width-1:0] bram_bm_addra;
wire [data_width-1:0] bram_bm_doutb;
wire [true_addr_width-1:0] bram_bm_addrb;

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_bm_addra_cnt <= 0;
    end
    else if (BM_top_gradx_out_valid & ~BM_top_gradx_out_last) begin
        bram_bm_addra_cnt <= bram_bm_addra_cnt + 1;
    end
    else if (BM_top_gradx_out_valid & BM_top_gradx_out_last) begin
        bram_bm_addra_cnt <= 0;
    end
end
assign bram_bm_addra = bram_bm_addra_cnt + bram_bm_addra_base;

always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        bram_bm_addra_base <= 0;
    end
    else if (BM_top_gradx_out_valid & BM_top_gradx_out_last) begin
        if (last_seq_bm) begin
            bram_bm_addra_base <= 0;
        end
        else begin
            bram_bm_addra_base <= bram_bm_addra_base + block_size;
        end
    end
end

reg [$clog2(sequence_length)-1:0] variance_rec_addr;
always @(posedge clk or negedge resetn) begin
    if (!resetn) begin
        variance_rec_addr <= 0;
    end
    else if (BM_top_gradx_out_valid & BM_top_gradx_out_last) begin
        if (last_seq_bm) begin
            variance_rec_addr <= 0;
        end
        else begin
            variance_rec_addr <= variance_rec_addr + 1;
        end
    end
end


assign bram_bm_addrb = mode_en[17] ? {{(true_addr_width-burst_addr_width){1'b0}}, axi_full_addr} : 0;
(* dont_touch = "yes" *) bram_gradx  u_bram_gradx(
    .addra  (bram_bm_addra),
    .clka   (clk),
    .dina   (BM_top_gradx_out),
    .douta  (),
    .wea    (BM_top_gradx_out_valid),

    .addrb  (bram_bm_addrb),
    .clkb   (clk),
    .dinb   (),
    .doutb  (bram_bm_doutb),
    .web    (0)
);

endmodule