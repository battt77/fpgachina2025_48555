
`timescale 1 ns / 1 ps

	module hfn_v3_0 #
	(
		// Users to add parameters here

		// User parameters ends
		// Do not modify the parameters beyond this line


		// Parameters of Axi Slave Bus Interface S00_AXI
		parameter integer C_S00_AXI_ID_WIDTH	= 1,
		parameter integer C_S00_AXI_DATA_WIDTH	= 128,
		parameter integer C_S00_AXI_ADDR_WIDTH	= 13,
		parameter integer C_S00_AXI_AWUSER_WIDTH	= 0,
		parameter integer C_S00_AXI_ARUSER_WIDTH	= 0,
		parameter integer C_S00_AXI_WUSER_WIDTH	= 0,
		parameter integer C_S00_AXI_RUSER_WIDTH	= 0,
		parameter integer C_S00_AXI_BUSER_WIDTH	= 0,

		// Parameters of Axi Slave Bus Interface S01_AXI
		parameter integer C_S01_AXI_DATA_WIDTH	= 32,
		parameter integer C_S01_AXI_ADDR_WIDTH	= 6
	)
	(
		// Users to add ports here
		input wire mode,
		output wire led,
		// User ports ends
		// Do not modify the ports beyond this line


		// Ports of Axi Slave Bus Interface S00_AXI
		input wire  s00_axi_aclk,
		input wire  s00_axi_aresetn,
		input wire [C_S00_AXI_ID_WIDTH-1 : 0] s00_axi_awid,
		input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_awaddr,
		input wire [7 : 0] s00_axi_awlen,
		input wire [2 : 0] s00_axi_awsize,
		input wire [1 : 0] s00_axi_awburst,
		input wire  s00_axi_awlock,
		input wire [3 : 0] s00_axi_awcache,
		input wire [2 : 0] s00_axi_awprot,
		input wire [3 : 0] s00_axi_awqos,
		input wire [3 : 0] s00_axi_awregion,
		input wire [C_S00_AXI_AWUSER_WIDTH-1 : 0] s00_axi_awuser,
		input wire  s00_axi_awvalid,
		output wire  s00_axi_awready,
		input wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_wdata,
		input wire [(C_S00_AXI_DATA_WIDTH/8)-1 : 0] s00_axi_wstrb,
		input wire  s00_axi_wlast,
		input wire [C_S00_AXI_WUSER_WIDTH-1 : 0] s00_axi_wuser,
		input wire  s00_axi_wvalid,
		output wire  s00_axi_wready,
		output wire [C_S00_AXI_ID_WIDTH-1 : 0] s00_axi_bid,
		output wire [1 : 0] s00_axi_bresp,
		output wire [C_S00_AXI_BUSER_WIDTH-1 : 0] s00_axi_buser,
		output wire  s00_axi_bvalid,
		input wire  s00_axi_bready,
		input wire [C_S00_AXI_ID_WIDTH-1 : 0] s00_axi_arid,
		input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_araddr,
		input wire [7 : 0] s00_axi_arlen,
		input wire [2 : 0] s00_axi_arsize,
		input wire [1 : 0] s00_axi_arburst,
		input wire  s00_axi_arlock,
		input wire [3 : 0] s00_axi_arcache,
		input wire [2 : 0] s00_axi_arprot,
		input wire [3 : 0] s00_axi_arqos,
		input wire [3 : 0] s00_axi_arregion,
		input wire [C_S00_AXI_ARUSER_WIDTH-1 : 0] s00_axi_aruser,
		input wire  s00_axi_arvalid,
		output wire  s00_axi_arready,
		output wire [C_S00_AXI_ID_WIDTH-1 : 0] s00_axi_rid,
		output wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_rdata,
		output wire [1 : 0] s00_axi_rresp,
		output wire  s00_axi_rlast,
		output wire [C_S00_AXI_RUSER_WIDTH-1 : 0] s00_axi_ruser,
		output wire  s00_axi_rvalid,
		input wire  s00_axi_rready,

		// Ports of Axi Slave Bus Interface S01_AXI
		input wire  s01_axi_aclk,
		input wire  s01_axi_aresetn,
		input wire [C_S01_AXI_ADDR_WIDTH-1 : 0] s01_axi_awaddr,
		input wire [2 : 0] s01_axi_awprot,
		input wire  s01_axi_awvalid,
		output wire  s01_axi_awready,
		input wire [C_S01_AXI_DATA_WIDTH-1 : 0] s01_axi_wdata,
		input wire [(C_S01_AXI_DATA_WIDTH/8)-1 : 0] s01_axi_wstrb,
		input wire  s01_axi_wvalid,
		output wire  s01_axi_wready,
		output wire [1 : 0] s01_axi_bresp,
		output wire  s01_axi_bvalid,
		input wire  s01_axi_bready,
		input wire [C_S01_AXI_ADDR_WIDTH-1 : 0] s01_axi_araddr,
		input wire [2 : 0] s01_axi_arprot,
		input wire  s01_axi_arvalid,
		output wire  s01_axi_arready,
		output wire [C_S01_AXI_DATA_WIDTH-1 : 0] s01_axi_rdata,
		output wire [1 : 0] s01_axi_rresp,
		output wire  s01_axi_rvalid,
		input wire  s01_axi_rready
	);
// Instantiation of Combined AXI Interface (AXI4-Full + AXI4-Lite + Norm_top)
	hfn_v3_0_AXI_COMBINED # ( 
		.C_S_AXIFULL_ID_WIDTH(C_S00_AXI_ID_WIDTH),
		.C_S_AXIFULL_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
		.C_S_AXIFULL_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH),
		.C_S_AXIFULL_AWUSER_WIDTH(C_S00_AXI_AWUSER_WIDTH),
		.C_S_AXIFULL_ARUSER_WIDTH(C_S00_AXI_ARUSER_WIDTH),
		.C_S_AXIFULL_WUSER_WIDTH(C_S00_AXI_WUSER_WIDTH),
		.C_S_AXIFULL_RUSER_WIDTH(C_S00_AXI_RUSER_WIDTH),
		.C_S_AXIFULL_BUSER_WIDTH(C_S00_AXI_BUSER_WIDTH),
		.C_S_AXILITE_DATA_WIDTH(C_S01_AXI_DATA_WIDTH),
		.C_S_AXILITE_ADDR_WIDTH(C_S01_AXI_ADDR_WIDTH)
	) hfn_v3_0_AXI_COMBINED_inst (
		// Global signals
		.ACLK(s00_axi_aclk),
		.ARESETN(s00_axi_aresetn),
		
		// AXI4-Full interface (S00_AXI)
		.S_AXIFULL_AWID(s00_axi_awid),
		.S_AXIFULL_AWADDR(s00_axi_awaddr),
		.S_AXIFULL_AWLEN(s00_axi_awlen),
		.S_AXIFULL_AWSIZE(s00_axi_awsize),
		.S_AXIFULL_AWBURST(s00_axi_awburst),
		.S_AXIFULL_AWLOCK(s00_axi_awlock),
		.S_AXIFULL_AWCACHE(s00_axi_awcache),
		.S_AXIFULL_AWPROT(s00_axi_awprot),
		.S_AXIFULL_AWQOS(s00_axi_awqos),
		.S_AXIFULL_AWREGION(s00_axi_awregion),
		.S_AXIFULL_AWUSER(s00_axi_awuser),
		.S_AXIFULL_AWVALID(s00_axi_awvalid),
		.S_AXIFULL_AWREADY(s00_axi_awready),
		.S_AXIFULL_WDATA(s00_axi_wdata),
		.S_AXIFULL_WSTRB(s00_axi_wstrb),
		.S_AXIFULL_WLAST(s00_axi_wlast),
		.S_AXIFULL_WUSER(s00_axi_wuser),
		.S_AXIFULL_WVALID(s00_axi_wvalid),
		.S_AXIFULL_WREADY(s00_axi_wready),
		.S_AXIFULL_BID(s00_axi_bid),
		.S_AXIFULL_BRESP(s00_axi_bresp),
		.S_AXIFULL_BUSER(s00_axi_buser),
		.S_AXIFULL_BVALID(s00_axi_bvalid),
		.S_AXIFULL_BREADY(s00_axi_bready),
		.S_AXIFULL_ARID(s00_axi_arid),
		.S_AXIFULL_ARADDR(s00_axi_araddr),
		.S_AXIFULL_ARLEN(s00_axi_arlen),
		.S_AXIFULL_ARSIZE(s00_axi_arsize),
		.S_AXIFULL_ARBURST(s00_axi_arburst),
		.S_AXIFULL_ARLOCK(s00_axi_arlock),
		.S_AXIFULL_ARCACHE(s00_axi_arcache),
		.S_AXIFULL_ARPROT(s00_axi_arprot),
		.S_AXIFULL_ARQOS(s00_axi_arqos),
		.S_AXIFULL_ARREGION(s00_axi_arregion),
		.S_AXIFULL_ARUSER(s00_axi_aruser),
		.S_AXIFULL_ARVALID(s00_axi_arvalid),
		.S_AXIFULL_ARREADY(s00_axi_arready),
		.S_AXIFULL_RID(s00_axi_rid),
		.S_AXIFULL_RDATA(s00_axi_rdata),
		.S_AXIFULL_RRESP(s00_axi_rresp),
		.S_AXIFULL_RLAST(s00_axi_rlast),
		.S_AXIFULL_RUSER(s00_axi_ruser),
		.S_AXIFULL_RVALID(s00_axi_rvalid),
		.S_AXIFULL_RREADY(s00_axi_rready),
		
		// AXI4-Lite interface (S01_AXI)
		.S_AXILITE_AWADDR(s01_axi_awaddr),
		.S_AXILITE_AWPROT(s01_axi_awprot),
		.S_AXILITE_AWVALID(s01_axi_awvalid),
		.S_AXILITE_AWREADY(s01_axi_awready),
		.S_AXILITE_WDATA(s01_axi_wdata),
		.S_AXILITE_WSTRB(s01_axi_wstrb),
		.S_AXILITE_WVALID(s01_axi_wvalid),
		.S_AXILITE_WREADY(s01_axi_wready),
		.S_AXILITE_BRESP(s01_axi_bresp),
		.S_AXILITE_BVALID(s01_axi_bvalid),
		.S_AXILITE_BREADY(s01_axi_bready),
		.S_AXILITE_ARADDR(s01_axi_araddr),
		.S_AXILITE_ARPROT(s01_axi_arprot),
		.S_AXILITE_ARVALID(s01_axi_arvalid),
		.S_AXILITE_ARREADY(s01_axi_arready),
		.S_AXILITE_RDATA(s01_axi_rdata),
		.S_AXILITE_RRESP(s01_axi_rresp),
		.S_AXILITE_RVALID(s01_axi_rvalid),
		.S_AXILITE_RREADY(s01_axi_rready),

		.mode			(mode)
	);

	// Add user logic here
	assign led = mode;
	// User logic ends

	endmodule
