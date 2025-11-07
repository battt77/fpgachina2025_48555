/*
 * Copyright (C) 2017 - 2019 Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <sleep.h>
#include "netif/xadapter.h"
#include "platform_config.h"
#include "xil_printf.h"
#include "lwip/init.h"
#include "lwip/inet.h"
#include "define.h"
#include "udp_func.h"
#include "app.h"
#include "xscugic.h"
#include "PL_data_manage.h"



static XScuGic IntcInstance;

#if LWIP_DHCP==1
#include "lwip/dhcp.h"
extern volatile int dhcp_timoutcntr;
err_t dhcp_start(struct netif *netif);
#endif

#ifdef XPS_BOARD_ZCU102
#ifdef XPAR_XIICPS_0_DEVICE_ID
int IicPhyReset(void);
#endif
#endif

static int complete_nw_thread;

void print_app_header();
void start_application();

#define THREAD_STACKSIZE 1024

struct netif server_netif;

static void print_ip(char *msg, ip_addr_t *ip)
{
	xil_printf(msg);
	xil_printf("%d.%d.%d.%d\n\r", ip4_addr1(ip), ip4_addr2(ip),
				ip4_addr3(ip), ip4_addr4(ip));
}

static void print_ip_settings(ip_addr_t *ip, ip_addr_t *mask, ip_addr_t *gw)
{
	print_ip("Board IP:       ", ip);
	print_ip("Netmask :       ", mask);
	print_ip("Gateway :       ", gw);
}

static void assign_default_ip(ip_addr_t *ip, ip_addr_t *mask, ip_addr_t *gw)
{
	int err;

	xil_printf("Configuring default IP %s \r\n", DEFAULT_IP_ADDRESS);

	err = inet_aton(DEFAULT_IP_ADDRESS, ip);
	if(!err)
		xil_printf("Invalid default IP address: %d\r\n", err);

	err = inet_aton(DEFAULT_IP_MASK, mask);
	if(!err)
		xil_printf("Invalid default IP MASK: %d\r\n", err);

	err = inet_aton(DEFAULT_GW_ADDRESS, gw);
	if(!err)
		xil_printf("Invalid default gateway address: %d\r\n", err);
}

void network_thread(void *p)
{
#if LWIP_DHCP==1
	int mscnt = 0;
#endif
	/* the mac address of the board. this should be unique per board */
	u8_t mac_ethernet_address[] = { 0x00, 0x0a, 0x35, 0x00, 0x01, 0x02 };

	/* Add network interface to the netif_list, and set it as default */
	if (!xemac_add(&server_netif, NULL, NULL, NULL, mac_ethernet_address,
		PLATFORM_EMAC_BASEADDR)) {
		xil_printf("Error adding N/W interface\r\n");
		return;
	}

	netif_set_default(&server_netif);

	/* specify that the network if is up */
	netif_set_up(&server_netif);

	/* start packet receive thread - required for lwIP operation */
	sys_thread_new("xemacif_input_thread",
			(void(*)(void*))xemacif_input_thread, &server_netif,
			THREAD_STACKSIZE, DEFAULT_THREAD_PRIO);

	complete_nw_thread = 1;

#if LWIP_DHCP==1
	dhcp_start(&server_netif);
	while (1) {
		vTaskDelay(DHCP_FINE_TIMER_MSECS / portTICK_RATE_MS);
		dhcp_fine_tmr();
		mscnt += DHCP_FINE_TIMER_MSECS;
		if (mscnt >= DHCP_COARSE_TIMER_SECS*1000) {
			dhcp_coarse_tmr();
			mscnt = 0;
		}
	}
#else
	vTaskDelete(NULL);
#endif
}

#if ENABLE_CDMA
static int setup_cdma_interrupt_only(XScuGic *IntcInstancePtr) {
    int Status;
    XScuGic_Config *IntcConfig;

    /* 
     * Setup interrupt controller for CDMA
     * Note: We initialize our own instance since lwIP may be using a different one
     */

    /* Get interrupt controller configuration */
    IntcConfig = XScuGic_LookupConfig(INTC_DEVICE_ID);
    if (NULL == IntcConfig) {
        xil_printf("[Error] Failed to lookup interrupt controller config\r\n");
        return XST_FAILURE;
    }

    /* Initialize interrupt controller for our use */
    Status = XScuGic_CfgInitialize(IntcInstancePtr, IntcConfig,
                                   IntcConfig->CpuBaseAddress);
    if (Status != XST_SUCCESS) {
        xil_printf("[Error] Failed to initialize interrupt controller (Status=%d)\r\n", Status);
        return XST_FAILURE;
    }
    xil_printf("  Interrupt controller initialized for CDMA\r\n");

    /* Connect CDMA interrupt handler */
    Status = XScuGic_Connect(IntcInstancePtr, CDMA_IRQ_ID,
                            (Xil_ExceptionHandler)cdma_isr,
                            (void *)IntcInstancePtr);
    if (Status != XST_SUCCESS) {
        xil_printf("[Error] Failed to connect CDMA interrupt (Status=%d)\r\n", Status);
        return XST_FAILURE;
    }

    /* Enable CDMA interrupt in GIC */
    XScuGic_Enable(IntcInstancePtr, CDMA_IRQ_ID);
    xil_printf("  CDMA interrupt connected and enabled (IRQ ID=%d)\r\n", CDMA_IRQ_ID);

    return XST_SUCCESS;
}

static int init_cdma_hw(void)
{
	u32 cr_value;
	u32 cr_readback;
	u32 sr;
	u32 timeout;

    xil_printf("\r\n=== Initializing CDMA (Polling Mode) ===\r\n");
    xil_printf("  CDMA Base Address = 0x%08x\r\n", CDMA_BASE);
    
    // 1. Check initial status
    sr = Xil_In32(CDMA_SR);
    cr_value = Xil_In32(CDMA_CR);
    xil_printf("  Initial CR = 0x%08x, SR = 0x%08x\r\n", cr_value, sr);
    
    // 2. Soft reset CDMA
    xil_printf("  Resetting CDMA...\r\n");
    Xil_Out32(CDMA_CR, CR_RESET);
    usleep(100);
    
    // 3. Wait for reset complete with timeout
    timeout = 10000;
    while ((Xil_In32(CDMA_CR) & CR_RESET) && timeout > 0) {
        usleep(1);
        timeout--;
    }
    if (timeout == 0) {
        cr_readback = Xil_In32(CDMA_CR);
        sr = Xil_In32(CDMA_SR);
        xil_printf("[Error] CDMA reset timeout! CR=0x%08x, SR=0x%08x\r\n", cr_readback, sr);
        xil_printf("[Error] CDMA may not exist at address 0x%08x\r\n", CDMA_BASE);
        return XST_FAILURE;
    }
    xil_printf("  CDMA reset complete (timeout remaining: %d)\r\n", timeout);

    // 4. Configure CDMA for POLLING mode (NO interrupts to avoid conflict with network stack)
    cr_value = 0x00000000;  // No interrupt enable bits set
    xil_printf("  Configuring CDMA for polling mode (CR = 0x%08x)\r\n", cr_value);
    Xil_Out32(CDMA_CR, cr_value);
    usleep(10);

    cr_readback = Xil_In32(CDMA_CR);
    xil_printf("  CR readback = 0x%08x\r\n", cr_readback);

    // 5. Clear any pending flags
    sr = Xil_In32(CDMA_SR);
    xil_printf("  SR = 0x%08x\r\n", sr);
    if (sr & (SR_IOC_IRQ | SR_ERR_IRQ)) {
        Xil_Out32(CDMA_SR, SR_IOC_IRQ | SR_ERR_IRQ);
        xil_printf("  Cleared status flags\r\n");
    }

    xil_printf("CDMA initialization complete (polling mode)!\r\n");
    xil_printf("Note: CDMA will use polling instead of interrupts to avoid conflict with network stack.\r\n\r\n");

	return XST_SUCCESS;
}

static int init_cdma_interrupt(void)
{
	int Status;
	
    xil_printf("\r\n=== Setting up CDMA Interrupt ===\r\n");
    
    // Setup CDMA interrupt (add to existing interrupt system)
    Status = setup_cdma_interrupt_only(&IntcInstance);
    if (Status != XST_SUCCESS) {
        xil_printf("[Error] Failed to setup CDMA interrupt!\r\n");
        return XST_FAILURE;
    }
    xil_printf("CDMA interrupt setup complete!\r\n\r\n");

	return XST_SUCCESS;
}
#endif  // ENABLE_CDMA



void main_app()
{
	// Main application loop
	xil_printf("Starting main application loop...\r\n");
	while(1)
	{
		app_main();
	}
}

int main_thread()
{
#if ENABLE_CDMA
	int Status;
#endif

#if LWIP_DHCP==1
	int mscnt = 0;
#endif

#ifdef XPS_BOARD_ZCU102
	IicPhyReset();
#endif
	xil_printf("\n\r\n\r");
	xil_printf("-----lwIP Socket Mode UDP Server Application------\r\n");

#if ENABLE_CDMA
	/* Step 1: Initialize CDMA hardware only (without interrupt setup) */
	Status = init_cdma_hw();
	if (Status != XST_SUCCESS) {
		xil_printf("[Error] CDMA hardware initialization failed!\r\n");
		vTaskDelete(NULL);
		return -1;
	}
#else
	xil_printf("[Info] CDMA is disabled (ENABLE_CDMA=0)\r\n");
#endif

	/* Step 2: Initialize lwIP (this will setup the interrupt system) */
	lwip_init();

	/* Step 3: Create network thread */
	sys_thread_new("nw_thread", network_thread, NULL,
			THREAD_STACKSIZE, DEFAULT_THREAD_PRIO);

	/* Step 4: Wait for network initialization complete */
	while(!complete_nw_thread)
		usleep(50);
	
	xil_printf("Network initialization complete\r\n");

#if LWIP_DHCP==1
	while (1) {
		vTaskDelay(DHCP_FINE_TIMER_MSECS / portTICK_RATE_MS);
		if (server_netif.ip_addr.addr) {
			xil_printf("DHCP request success\r\n");
			break;
		}
		mscnt += DHCP_FINE_TIMER_MSECS;
		if (mscnt >= 10000) {
			xil_printf("ERROR: DHCP request timed out\r\n");
			assign_default_ip(&(server_netif.ip_addr),
						&(server_netif.netmask),
						&(server_netif.gw));
			break;
		}
	}

#else
	assign_default_ip(&(server_netif.ip_addr), &(server_netif.netmask),
				&(server_netif.gw));
#endif

	print_ip_settings(&(server_netif.ip_addr), &(server_netif.netmask),
				&(server_netif.gw));
	xil_printf("\r\n");

#if ENABLE_CDMA
	/* CDMA is configured in polling mode - no interrupt setup needed */
	xil_printf("[Info] CDMA is running in polling mode (no interrupts)\r\n");
#endif

	/* print all application headers */
	print_app_header();
	xil_printf("\r\n");

	/* start the application*/
	start_application();

//	/* Create main application thread */
//	sys_thread_new("main_app", (void(*)(void*))main_app, NULL,
//			THREAD_STACKSIZE, DEFAULT_THREAD_PRIO);

	vTaskDelete(NULL);
	return 0;
}


int main()
{
	/* Create main initialization thread */
	sys_thread_new("main_thread", (void(*)(void*))main_thread, 0,
			THREAD_STACKSIZE, DEFAULT_THREAD_PRIO);

	sys_thread_new("main_app", (void(*)(void*))main_app, 0,
			THREAD_STACKSIZE, DEFAULT_THREAD_PRIO);

	/* Start FreeRTOS scheduler */
	vTaskStartScheduler();
	
	/* Should never reach here */
	while(1);
	return 0;
}
