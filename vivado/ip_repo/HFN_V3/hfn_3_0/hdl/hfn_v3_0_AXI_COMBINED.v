
`timescale 1 ns / 1 ps

module hfn_v3_0_AXI_COMBINED #
(
	// Users to add parameters here

	// User parameters ends
	// Do not modify the parameters beyond this line

	// Parameters for AXI4-Full interface (Data Stream)
	parameter integer C_S_AXIFULL_ID_WIDTH	= 1,
	parameter integer C_S_AXIFULL_DATA_WIDTH	= 128,
	parameter integer C_S_AXIFULL_ADDR_WIDTH	= 13,  // 13 bits for 256-burst addressing (5 LSB + 8 address bits)
	parameter integer C_S_AXIFULL_AWUSER_WIDTH	= 0,
	parameter integer C_S_AXIFULL_ARUSER_WIDTH	= 0,
	parameter integer C_S_AXIFULL_WUSER_WIDTH	= 0,
	parameter integer C_S_AXIFULL_RUSER_WIDTH	= 0,
	parameter integer C_S_AXIFULL_BUSER_WIDTH	= 0,

	// Parameters for AXI4-Lite interface (Control Registers)
	parameter integer C_S_AXILITE_DATA_WIDTH	= 32,
	parameter integer C_S_AXILITE_ADDR_WIDTH	= 6  // 6 bits for 10 registers (4 bits for reg select + 2 bits for byte addressing)
)
(
	// Users to add ports here

	// User ports ends
	// Do not modify the ports beyond this line

	// Global Clock Signal (shared by both interfaces)
	input wire  ACLK,
	// Global Reset Signal. This Signal is Active LOW (shared by both interfaces)
	input wire  ARESETN,

	// ============================================
	// AXI4-Full Interface Ports (Data Stream)
	// ============================================
	
	// Write Address Channel
	input wire [C_S_AXIFULL_ID_WIDTH-1 : 0] S_AXIFULL_AWID,
	input wire [C_S_AXIFULL_ADDR_WIDTH-1 : 0] S_AXIFULL_AWADDR,
	input wire [7 : 0] S_AXIFULL_AWLEN,
	input wire [2 : 0] S_AXIFULL_AWSIZE,
	input wire [1 : 0] S_AXIFULL_AWBURST,
	input wire  S_AXIFULL_AWLOCK,
	input wire [3 : 0] S_AXIFULL_AWCACHE,
	input wire [2 : 0] S_AXIFULL_AWPROT,
	input wire [3 : 0] S_AXIFULL_AWQOS,
	input wire [3 : 0] S_AXIFULL_AWREGION,
	input wire [C_S_AXIFULL_AWUSER_WIDTH-1 : 0] S_AXIFULL_AWUSER,
	input wire  S_AXIFULL_AWVALID,
	output wire  S_AXIFULL_AWREADY,
	
	// Write Data Channel
	input wire [C_S_AXIFULL_DATA_WIDTH-1 : 0] S_AXIFULL_WDATA,
	input wire [(C_S_AXIFULL_DATA_WIDTH/8)-1 : 0] S_AXIFULL_WSTRB,
	input wire  S_AXIFULL_WLAST,
	input wire [C_S_AXIFULL_WUSER_WIDTH-1 : 0] S_AXIFULL_WUSER,
	input wire  S_AXIFULL_WVALID,
	output wire  S_AXIFULL_WREADY,
	
	// Write Response Channel
	output wire [C_S_AXIFULL_ID_WIDTH-1 : 0] S_AXIFULL_BID,
	output wire [1 : 0] S_AXIFULL_BRESP,
	output wire [C_S_AXIFULL_BUSER_WIDTH-1 : 0] S_AXIFULL_BUSER,
	output wire  S_AXIFULL_BVALID,
	input wire  S_AXIFULL_BREADY,
	
	// Read Address Channel
	input wire [C_S_AXIFULL_ID_WIDTH-1 : 0] S_AXIFULL_ARID,
	input wire [C_S_AXIFULL_ADDR_WIDTH-1 : 0] S_AXIFULL_ARADDR,
	input wire [7 : 0] S_AXIFULL_ARLEN,
	input wire [2 : 0] S_AXIFULL_ARSIZE,
	input wire [1 : 0] S_AXIFULL_ARBURST,
	input wire  S_AXIFULL_ARLOCK,
	input wire [3 : 0] S_AXIFULL_ARCACHE,
	input wire [2 : 0] S_AXIFULL_ARPROT,
	input wire [3 : 0] S_AXIFULL_ARQOS,
	input wire [3 : 0] S_AXIFULL_ARREGION,
	input wire [C_S_AXIFULL_ARUSER_WIDTH-1 : 0] S_AXIFULL_ARUSER,
	input wire  S_AXIFULL_ARVALID,
	output wire  S_AXIFULL_ARREADY,
	
	// Read Data Channel
	output wire [C_S_AXIFULL_ID_WIDTH-1 : 0] S_AXIFULL_RID,
	output wire [C_S_AXIFULL_DATA_WIDTH-1 : 0] S_AXIFULL_RDATA,
	output wire [1 : 0] S_AXIFULL_RRESP,
	output wire  S_AXIFULL_RLAST,
	output wire [C_S_AXIFULL_RUSER_WIDTH-1 : 0] S_AXIFULL_RUSER,
	output wire  S_AXIFULL_RVALID,
	input wire  S_AXIFULL_RREADY,

	// ============================================
	// AXI4-Lite Interface Ports (Control Registers)
	// ============================================
	
	// Write Address Channel
	input wire [C_S_AXILITE_ADDR_WIDTH-1 : 0] S_AXILITE_AWADDR,
	input wire [2 : 0] S_AXILITE_AWPROT,
	input wire  S_AXILITE_AWVALID,
	output wire  S_AXILITE_AWREADY,
	
	// Write Data Channel
	input wire [C_S_AXILITE_DATA_WIDTH-1 : 0] S_AXILITE_WDATA,
	input wire [(C_S_AXILITE_DATA_WIDTH/8)-1 : 0] S_AXILITE_WSTRB,
	input wire  S_AXILITE_WVALID,
	output wire  S_AXILITE_WREADY,
	
	// Write Response Channel
	output wire [1 : 0] S_AXILITE_BRESP,
	output wire  S_AXILITE_BVALID,
	input wire  S_AXILITE_BREADY,
	
	// Read Address Channel
	input wire [C_S_AXILITE_ADDR_WIDTH-1 : 0] S_AXILITE_ARADDR,
	input wire [2 : 0] S_AXILITE_ARPROT,
	input wire  S_AXILITE_ARVALID,
	output wire  S_AXILITE_ARREADY,
	
	// Read Data Channel
	output wire [C_S_AXILITE_DATA_WIDTH-1 : 0] S_AXILITE_RDATA,
	output wire [1 : 0] S_AXILITE_RRESP,
	output wire  S_AXILITE_RVALID,
	input wire  S_AXILITE_RREADY,

	input wire mode
);

	// ============================================
	// AXI4-Full Signals (Data Stream)
	// ============================================
	reg [C_S_AXIFULL_ADDR_WIDTH-1 : 0] 	axifull_awaddr;
	reg  	axifull_awready;
	reg  	axifull_wready;
	reg [1 : 0] 	axifull_bresp;
	reg [C_S_AXIFULL_BUSER_WIDTH-1 : 0] 	axifull_buser;
	reg  	axifull_bvalid;
	reg [C_S_AXIFULL_ADDR_WIDTH-1 : 0] 	axifull_araddr;
	reg  	axifull_arready;
	reg [C_S_AXIFULL_DATA_WIDTH-1 : 0] 	axifull_rdata;
	reg [1 : 0] 	axifull_rresp;
	reg  	axifull_rlast;
	reg [C_S_AXIFULL_RUSER_WIDTH-1 : 0] 	axifull_ruser;
	reg  	axifull_rvalid;
	
	wire axifull_aw_wrap_en;
	wire axifull_ar_wrap_en;
	wire [31:0]  axifull_aw_wrap_size ; 
	wire [31:0]  axifull_ar_wrap_size ; 
	reg axifull_awv_awr_flag;
	reg axifull_arv_arr_flag; 
	reg [7:0] axifull_awlen_cntr;
	reg [7:0] axifull_arlen_cntr;
	reg [1:0] axifull_arburst;
	reg [1:0] axifull_awburst;
	reg [7:0] axifull_arlen;
	reg [7:0] axifull_awlen;
	
	localparam integer AXIFULL_ADDR_LSB = (C_S_AXIFULL_DATA_WIDTH/32)+ 1;
	localparam integer AXIFULL_OPT_MEM_ADDR_BITS = 8;  // 8 bits for 256 addresses - 支持4*512突发传输
	
	wire [AXIFULL_OPT_MEM_ADDR_BITS-1:0] axifull_mem_address;
	wire [C_S_AXIFULL_DATA_WIDTH-1:0] bram_porta_dout;

	// ============================================
	// AXI4-Lite Signals (Control Registers)
	// ============================================
	reg [C_S_AXILITE_ADDR_WIDTH-1 : 0] 	axi_awaddr;
	reg  	axi_awready;
	reg  	axi_wready;
	reg [1 : 0] 	axi_bresp;
	reg  	axi_bvalid;
	reg [C_S_AXILITE_ADDR_WIDTH-1 : 0] 	axi_araddr;
	reg  	axi_arready;
	reg [C_S_AXILITE_DATA_WIDTH-1 : 0] 	axi_rdata;
	reg [1 : 0] 	axi_rresp;
	reg  	axi_rvalid;

	localparam integer ADDR_LSB = (C_S_AXILITE_DATA_WIDTH/32) + 1;
	localparam integer OPT_MEM_ADDR_BITS = 3;  // 4 bits for 10 registers
	
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg0;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg1;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg2;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg3;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg4;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg5;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg6;
	wire [C_S_AXILITE_DATA_WIDTH-1:0]	status_reg0;  // Read-only status register from Norm_top
	wire [C_S_AXILITE_DATA_WIDTH-1:0]	status_reg1;  // Read-only status register from Norm_top
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg7;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg8;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	slv_reg9;
	wire	 slv_reg_rden;
	wire	 slv_reg_wren;
	reg [C_S_AXILITE_DATA_WIDTH-1:0]	 reg_data_out;
	integer	 byte_index;
	reg	 aw_en;

	// ============================================
	// AXI4-Full I/O Connections assignments
	// ============================================
	assign S_AXIFULL_AWREADY	= axifull_awready;
	assign S_AXIFULL_WREADY	= axifull_wready;
	assign S_AXIFULL_BRESP	= axifull_bresp;
	assign S_AXIFULL_BUSER	= axifull_buser;
	assign S_AXIFULL_BVALID	= axifull_bvalid;
	assign S_AXIFULL_ARREADY	= axifull_arready;
	assign S_AXIFULL_RDATA	= axifull_rdata;
	assign S_AXIFULL_RRESP	= axifull_rresp;
	assign S_AXIFULL_RLAST	= axifull_rlast;
	assign S_AXIFULL_RUSER	= axifull_ruser;
	assign S_AXIFULL_RVALID	= axifull_rvalid;
	assign S_AXIFULL_BID = S_AXIFULL_AWID;
	assign S_AXIFULL_RID = S_AXIFULL_ARID;
	assign  axifull_aw_wrap_size = (C_S_AXIFULL_DATA_WIDTH/8 * (axifull_awlen)); 
	assign  axifull_ar_wrap_size = (C_S_AXIFULL_DATA_WIDTH/8 * (axifull_arlen)); 
	assign  axifull_aw_wrap_en = ((axifull_awaddr & axifull_aw_wrap_size) == axifull_aw_wrap_size)? 1'b1: 1'b0;
	assign  axifull_ar_wrap_en = ((axifull_araddr & axifull_ar_wrap_size) == axifull_ar_wrap_size)? 1'b1: 1'b0;

	// ============================================
	// AXI4-Lite I/O Connections assignments
	// ============================================
	assign S_AXILITE_AWREADY	= axi_awready;
	assign S_AXILITE_WREADY	= axi_wready;
	assign S_AXILITE_BRESP	= axi_bresp;
	assign S_AXILITE_BVALID	= axi_bvalid;
	assign S_AXILITE_ARREADY	= axi_arready;
	assign S_AXILITE_RDATA	= axi_rdata;
	assign S_AXILITE_RRESP	= axi_rresp;
	assign S_AXILITE_RVALID	= axi_rvalid;

	// ============================================
	// AXI4-Full Implementation (Data Stream)
	// ============================================

	// Implement axifull_awready generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axifull_awready <= 1'b0;
	      axifull_awv_awr_flag <= 1'b0;
	    end 
	  else
	    begin    
	      if (~axifull_awready && S_AXIFULL_AWVALID && ~axifull_awv_awr_flag && ~axifull_arv_arr_flag)
	        begin
	          axifull_awready <= 1'b1;
	          axifull_awv_awr_flag  <= 1'b1; 
	        end
	      else if (S_AXIFULL_WLAST && axifull_wready)          
	        begin
	          axifull_awv_awr_flag  <= 1'b0;
	        end
	      else        
	        begin
	          axifull_awready <= 1'b0;
	        end
	    end 
	end       

	// Implement axifull_awaddr latching
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axifull_awaddr <= 0;
	      axifull_awlen_cntr <= 0;
	      axifull_awburst <= 0;
	      axifull_awlen <= 0;
	    end 
	  else
	    begin    
	      if (~axifull_awready && S_AXIFULL_AWVALID && ~axifull_awv_awr_flag)
	        begin
	          axifull_awaddr <= S_AXIFULL_AWADDR[C_S_AXIFULL_ADDR_WIDTH - 1:0];  
	          axifull_awburst <= S_AXIFULL_AWBURST; 
	          axifull_awlen <= S_AXIFULL_AWLEN;     
	          axifull_awlen_cntr <= 0;
	        end   
	      else if((axifull_awlen_cntr <= axifull_awlen) && axifull_wready && S_AXIFULL_WVALID)        
	        begin
	          axifull_awlen_cntr <= axifull_awlen_cntr + 1;

	          case (axifull_awburst)
	            2'b00: // fixed burst
	              begin
	                axifull_awaddr <= axifull_awaddr;          
	              end   
	            2'b01: //incremental burst
	              begin
	                axifull_awaddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] <= axifull_awaddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] + 1;
	                axifull_awaddr[AXIFULL_ADDR_LSB-1:0]  <= {AXIFULL_ADDR_LSB{1'b0}};   
	              end   
	            2'b10: //Wrapping burst
	              if (axifull_aw_wrap_en)
	                begin
	                  axifull_awaddr <= (axifull_awaddr - axifull_aw_wrap_size); 
	                end
	              else 
	                begin
	                  axifull_awaddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] <= axifull_awaddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] + 1;
	                  axifull_awaddr[AXIFULL_ADDR_LSB-1:0]  <= {AXIFULL_ADDR_LSB{1'b0}}; 
	                end                      
	            default: //reserved (incremental burst for example)
	              begin
	                axifull_awaddr <= axifull_awaddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] + 1;
	              end
	          endcase              
	        end
	    end 
	end       

	// Implement axifull_wready generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axifull_wready <= 1'b0;
	    end 
	  else
	    begin    
	      if ( ~axifull_wready && S_AXIFULL_WVALID && axifull_awv_awr_flag)
	        begin
	          axifull_wready <= 1'b1;
	        end
	      else if (S_AXIFULL_WLAST && axifull_wready)
	        begin
	          axifull_wready <= 1'b0;
	        end
	    end 
	end       

	// Implement write response logic generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axifull_bvalid <= 0;
	      axifull_bresp <= 2'b0;
	      axifull_buser <= 0;
	    end 
	  else
	    begin    
	      if (axifull_awv_awr_flag && axifull_wready && S_AXIFULL_WVALID && ~axifull_bvalid && S_AXIFULL_WLAST )
	        begin
	          axifull_bvalid <= 1'b1;
	          axifull_bresp  <= 2'b0; 
	        end                   
	      else
	        begin
	          if (S_AXIFULL_BREADY && axifull_bvalid) 
	            begin
	              axifull_bvalid <= 1'b0; 
	            end  
	        end
	    end
	 end   

	// Implement axifull_arready generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axifull_arready <= 1'b0;
	      axifull_arv_arr_flag <= 1'b0;
	    end 
	  else
	    begin    
	      if (~axifull_arready && S_AXIFULL_ARVALID && ~axifull_awv_awr_flag && ~axifull_arv_arr_flag)
	        begin
	          axifull_arready <= 1'b1;
	          axifull_arv_arr_flag <= 1'b1;
	        end
	      else if (axifull_rvalid && S_AXIFULL_RREADY && axifull_arlen_cntr == axifull_arlen)
	        begin
	          axifull_arv_arr_flag  <= 1'b0;
	        end
	      else        
	        begin
	          axifull_arready <= 1'b0;
	        end
	    end 
	end       

	// Implement axifull_araddr latching
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axifull_araddr <= 0;
	      axifull_arlen_cntr <= 0;
	      axifull_arburst <= 0;
	      axifull_arlen <= 0;
	      axifull_rlast <= 1'b0;
	      axifull_ruser <= 0;
	    end 
	  else
	    begin    
	      if (~axifull_arready && S_AXIFULL_ARVALID && ~axifull_arv_arr_flag)
	        begin
	          axifull_araddr <= S_AXIFULL_ARADDR[C_S_AXIFULL_ADDR_WIDTH - 1:0]; 
	          axifull_arburst <= S_AXIFULL_ARBURST; 
	          axifull_arlen <= S_AXIFULL_ARLEN;     
	          axifull_arlen_cntr <= 0;
	          axifull_rlast <= 1'b0;
	        end   
	      else if((axifull_arlen_cntr <= axifull_arlen) && axifull_rvalid && S_AXIFULL_RREADY)        
	        begin
	          axifull_arlen_cntr <= axifull_arlen_cntr + 1;
	          axifull_rlast <= 1'b0;
	        
	          case (axifull_arburst)
	            2'b00: // fixed burst
	              begin
	                axifull_araddr       <= axifull_araddr;        
	              end   
	            2'b01: //incremental burst
	              begin
	                axifull_araddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] <= axifull_araddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] + 1; 
	                axifull_araddr[AXIFULL_ADDR_LSB-1:0]  <= {AXIFULL_ADDR_LSB{1'b0}};   
	              end   
	            2'b10: //Wrapping burst
	              if (axifull_ar_wrap_en) 
	                begin
	                  axifull_araddr <= (axifull_araddr - axifull_ar_wrap_size); 
	                end
	              else 
	                begin
	                axifull_araddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] <= axifull_araddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB] + 1; 
	                axifull_araddr[AXIFULL_ADDR_LSB-1:0]  <= {AXIFULL_ADDR_LSB{1'b0}};   
	                end                      
	            default: //reserved (incremental burst for example)
	              begin
	                axifull_araddr <= axifull_araddr[C_S_AXIFULL_ADDR_WIDTH - 1:AXIFULL_ADDR_LSB]+1;
	              end
	          endcase              
	        end
	      else if((axifull_arlen_cntr == axifull_arlen) && ~axifull_rlast && axifull_arv_arr_flag )   
	        begin
	          axifull_rlast <= 1'b1;
	        end          
	      else if (S_AXIFULL_RREADY)   
	        begin
	          axifull_rlast <= 1'b0;
	        end          
	    end 
	end       

	// Implement axifull_rvalid generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axifull_rvalid <= 0;
	      axifull_rresp  <= 0;
	    end 
	  else
	    begin    
	      if (axifull_arv_arr_flag && ~axifull_rvalid)
	        begin
	          axifull_rvalid <= 1'b1;
	          axifull_rresp  <= 2'b0; 
	        end   
	      else if (axifull_rvalid && S_AXIFULL_RREADY)
	        begin
	          axifull_rvalid <= 1'b0;
	        end            
	    end
	end    

	// ============================================
	// AXI4-Full BRAM Interface
	// ============================================
	
	// Address generation for BRAM
	assign axifull_mem_address = (axifull_arv_arr_flag? 
	                               axifull_araddr[AXIFULL_ADDR_LSB+AXIFULL_OPT_MEM_ADDR_BITS-1:AXIFULL_ADDR_LSB] :
	                               (axifull_awv_awr_flag? 
	                                axifull_awaddr[AXIFULL_ADDR_LSB+AXIFULL_OPT_MEM_ADDR_BITS-1:AXIFULL_ADDR_LSB] : 
	                                {AXIFULL_OPT_MEM_ADDR_BITS{1'b0}}));
	
	// AXI-Full to Norm_top control signals
	wire bram_porta_we;
	
	assign bram_porta_we = (axifull_wready && S_AXIFULL_WVALID);
	
	// Output memory read data (BRAM already has 1 cycle read latency)
	always @(*)
	begin
	  if (axifull_rvalid) 
	    begin
	      axifull_rdata = bram_porta_dout;
	    end   
	  else
	    begin
	      axifull_rdata = {C_S_AXIFULL_DATA_WIDTH{1'b0}};
	    end       
	end    

	// ============================================
	// AXI4-Lite Implementation (Control Registers)
	// ============================================

	// Implement axi_awready generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axi_awready <= 1'b0;
	      aw_en <= 1'b1;
	    end 
	  else
	    begin    
	      if (~axi_awready && S_AXILITE_AWVALID && S_AXILITE_WVALID && aw_en)
	        begin
	          axi_awready <= 1'b1;
	          aw_en <= 1'b0;
	        end
	        else if (S_AXILITE_BREADY && axi_bvalid)
	            begin
	              aw_en <= 1'b1;
	              axi_awready <= 1'b0;
	            end
	      else           
	        begin
	          axi_awready <= 1'b0;
	        end
	    end 
	end       

	// Implement axi_awaddr latching
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axi_awaddr <= 0;
	    end 
	  else
	    begin    
	      if (~axi_awready && S_AXILITE_AWVALID && S_AXILITE_WVALID && aw_en)
	        begin
	          axi_awaddr <= S_AXILITE_AWADDR;
	        end
	    end 
	end       

	// Implement axi_wready generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axi_wready <= 1'b0;
	    end 
	  else
	    begin    
	      if (~axi_wready && S_AXILITE_WVALID && S_AXILITE_AWVALID && aw_en )
	        begin
	          axi_wready <= 1'b1;
	        end
	      else
	        begin
	          axi_wready <= 1'b0;
	        end
	    end 
	end       

	// Implement memory mapped register select and write logic generation
	assign slv_reg_wren = axi_wready && S_AXILITE_WVALID && axi_awready && S_AXILITE_AWVALID;

	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      slv_reg0 <= 0;
	      slv_reg1 <= 0;
	      slv_reg2 <= 0;
	      slv_reg3 <= 0;
	      slv_reg4 <= 0;
		  slv_reg5 <= 0;
		  slv_reg6 <= 0;
	      slv_reg7 <= 0;
	      slv_reg8 <= 0;
	      slv_reg9 <= 0;
	    end 
	  else begin
	    if (slv_reg_wren)
	      begin
	        case ( axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] )
	          4'h0:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg0[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  
	          4'h1:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg1[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  
	          4'h2:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg2[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  
	          4'h3:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg3[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  
	          4'h4:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg4[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  

				4'h5:
					for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
					if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
						// Respective byte enables are asserted as per write strobes 
						// Slave register 5
						slv_reg5[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
					end  
				4'h6:
					for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
					if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
						// Respective byte enables are asserted as per write strobes 
						// Slave register 6
						slv_reg6[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
					end  
	          4'h7:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg7[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  
	          4'h8:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg8[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  
	          4'h9:
	            for ( byte_index = 0; byte_index <= (C_S_AXILITE_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXILITE_WSTRB[byte_index] == 1 ) begin
	                slv_reg9[(byte_index*8) +: 8] <= S_AXILITE_WDATA[(byte_index*8) +: 8];
	              end  
	          default : begin
	                      slv_reg0 <= slv_reg0;
	                      slv_reg1 <= slv_reg1;
	                      slv_reg2 <= slv_reg2;
	                      slv_reg3 <= slv_reg3;
	                      slv_reg4 <= slv_reg4;
						  slv_reg5 <= slv_reg5;
						  slv_reg6 <= slv_reg6;
	                      slv_reg7 <= slv_reg7;
	                      slv_reg8 <= slv_reg8;
	                      slv_reg9 <= slv_reg9;
	                    end
	        endcase
	      end
	  end
	end    

	// Implement write response logic generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axi_bvalid  <= 0;
	      axi_bresp   <= 2'b0;
	    end 
	  else
	    begin    
	      if (axi_awready && S_AXILITE_AWVALID && ~axi_bvalid && axi_wready && S_AXILITE_WVALID)
	        begin
	          axi_bvalid <= 1'b1;
	          axi_bresp  <= 2'b0;
	        end                   
	      else
	        begin
	          if (S_AXILITE_BREADY && axi_bvalid) 
	            begin
	              axi_bvalid <= 1'b0; 
	            end  
	        end
	    end
	end   

	// Implement axi_arready generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axi_arready <= 1'b0;
	      axi_araddr  <= {C_S_AXILITE_ADDR_WIDTH{1'b0}};
	    end 
	  else
	    begin    
	      if (~axi_arready && S_AXILITE_ARVALID)
	        begin
	          axi_arready <= 1'b1;
	          axi_araddr  <= S_AXILITE_ARADDR;
	        end
	      else
	        begin
	          axi_arready <= 1'b0;
	        end
	    end 
	end       

	// Implement axi_rvalid generation
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axi_rvalid <= 0;
	      axi_rresp  <= 0;
	    end 
	  else
	    begin    
	      if (axi_arready && S_AXILITE_ARVALID && ~axi_rvalid)
	        begin
	          axi_rvalid <= 1'b1;
	          axi_rresp  <= 2'b0;
	        end   
	      else if (axi_rvalid && S_AXILITE_RREADY)
	        begin
	          axi_rvalid <= 1'b0;
	        end                
	    end
	end    

	// Implement memory mapped register select and read logic generation
	assign slv_reg_rden = axi_arready & S_AXILITE_ARVALID & ~axi_rvalid;
	
	always @(*)
	begin
	      case ( axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] )
	        4'h0   : reg_data_out <= slv_reg0;
	        4'h1   : reg_data_out <= slv_reg1;
	        4'h2   : reg_data_out <= slv_reg2;
	        4'h3   : reg_data_out <= slv_reg3;
	        4'h4   : reg_data_out <= slv_reg4;
	        4'h5   : reg_data_out <= status_reg0;
	        4'h6   : reg_data_out <= status_reg1;
	        4'h7   : reg_data_out <= bm_param_result;
	        4'h8   : reg_data_out <= tst1;
	        4'h9   : reg_data_out <= tst2;
	        default : reg_data_out <= 0;
	      endcase
	end

	// Output register or memory read data
	always @( posedge ACLK )
	begin
	  if ( ARESETN == 1'b0 )
	    begin
	      axi_rdata  <= 0;
	    end 
	  else
	    begin    
	      if (slv_reg_rden)
	        begin
	          axi_rdata <= reg_data_out;
	        end   
	    end
	end    

	// ============================================
	// User logic here - Norm_top instantiation
	// ============================================
	wire [31:0] bm_param_result;
	wire [31:0] tst1;
	wire [31:0] tst2;
	Norm_top #(
		.N                 (8),
		.data_width        (C_S_AXIFULL_DATA_WIDTH),
		.burst_addr_width  (AXIFULL_OPT_MEM_ADDR_BITS),
		.sequence_length   (4),
		.hidden_dim        (512)
	) u_Norm_top (
		.clk          (ACLK),
		.resetn       (ARESETN),
		.mode			(mode),
		
		// AXI-Full interface
		.axi_full_addr   (axifull_mem_address),
		.axi_full_din    (S_AXIFULL_WDATA),
		.axi_full_dout   (bram_porta_dout),
		.axi_full_we     (bram_porta_we),
		
		// Control/Status signals from AXI-Lite
		.ctrl_reg0    (slv_reg0),
		.ctrl_reg1    (slv_reg1),
		.constant0    (slv_reg2),
		.constant1    (slv_reg3),
		.constant2    (slv_reg4),
		.status_reg0  (status_reg0),
		.status_reg1  (status_reg1),
		.bm_param_result (bm_param_result),
		.tst1         	 (tst1),
		.tst2            (tst2)
	);
	
	// User logic ends

endmodule

