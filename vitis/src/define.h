#ifndef __DEFINE_
#define __DEFINE_

#define float_bitwidth  16
#define seq 	4
#define hidden  512
#define N	    8
#define H       hidden
#define E       1e-6f
#define total_B 4096

#include <stdint.h>
#include <xbasic_types.h>

typedef uint16_t fp;

#include "xparameters.h"

//local ip
#define DEFAULT_IP_ADDRESS	"192.168.1.10"
#define DEFAULT_IP_MASK		"255.255.255.0"
#define DEFAULT_GW_ADDRESS	"192.168.1.1"
/* Server to connect with */
#define UDP_REMOTE_IP_ADDRESS "192.168.1.20"

/* server port to listen on/connect to */
#define UDP_CONN_PORT 5001
#define UDP_REMOTE_PORT 8080

//DDR addr in manage
#define XDMA_DDR_BASE_ADDR			0x00000000
#define PS_DDR_OFFSET				0x500000000  // PS端访问DDR偏移
#define SEQ_OFFSET					0x400        // sequence间偏移 (1024字节 = 512*2 = 1KB)

#define	X_BASE_ADDR				(XDMA_DDR_BASE_ADDR) //21474836480
#define	WEIGHT_BASE_ADDR		(XDMA_DDR_BASE_ADDR+0x00100000) //21475885056
#define BIAS_BASE_ADDR			(XDMA_DDR_BASE_ADDR+0x00200000) //21476933632
#define DELTA_BASE_ADDR			(XDMA_DDR_BASE_ADDR+0x00300000) //21477982208

//DDR result
#define NORM_ADDR				(XDMA_DDR_BASE_ADDR+0x00400000) //21479030784
#define FW_RESULT_ADDR			(XDMA_DDR_BASE_ADDR+0x00500000)
#define BW_X_RESULT_ADDR		(XDMA_DDR_BASE_ADDR+0x00600000)
#define BW_W_RESULT_ADDR		(XDMA_DDR_BASE_ADDR+0x00700000)
#define BW_B_RESULT_ADDR		(XDMA_DDR_BASE_ADDR+0x00800000)

//XDMA PL AXI FULL ADDR
#define RLN_BASE_AXIF 0xB0000000

//DDR result
#define CAL_CLK_ADDR            (XDMA_DDR_BASE_ADDR+0x00900000)    

//AXI-Lite ADDR
#define RLN_BASE XPAR_HFN_0_S01_AXI_BASEADDR
#define reg0 0
#define reg1 4
#define reg2 8
#define reg3 12
#define reg4 16
#define reg5 20
#define reg6 24
#define reg7 28
#define reg8 32


#define INTC_DEVICE_ID      XPAR_SCUGIC_0_DEVICE_ID

// Enable/Disable CDMA (set to 0 if CDMA is not in hardware design)
#define ENABLE_CDMA         1

#if ENABLE_CDMA
#define CDMA_IRQ_ID         XPAR_FABRIC_AXICDMA_0_VEC_ID
#define CDMA_BASE           XPAR_AXICDMA_0_BASEADDR
#define CDMA_CR     (CDMA_BASE + 0x00)   // Control
#define CDMA_SR     (CDMA_BASE + 0x04)   // Status
#define CDMA_SA     (CDMA_BASE + 0x18)   // Source Address
#define CDMA_DA     (CDMA_BASE + 0x20)   // Destination Address
#define CDMA_BTT    (CDMA_BASE + 0x28)   // Bytes To Transfer

// CDMA Control Register bits
#define CR_RESET        0x00000004       // Soft reset
#define CR_IOC_IRQEN    0x00001000       // Interrupt on Complete Enable
#define CR_ERR_IRQEN    0x00004000       // Error Interrupt Enable

// CDMA Status Register bits
#define SR_IDLE         0x00000002       // CDMA is idle
#define SR_IOC_IRQ      0x00001000       // Interrupt on Complete
#define SR_ERR_IRQ      0x00004000       // Error Interrupt
#endif  // ENABLE_CDMA

#endif
