#include  "PL_data_manage.h"
#include  "app.h"
#include  "xil_io.h"

extern u32 PL_feedback;

static uint16_t fp32_to_fp16(float value) {
    uint32_t f32 = *(uint32_t*)&value;
    
    uint32_t sign = (f32 >> 31) & 0x1;
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t frac = f32 & 0x7FFFFF;
    
    uint16_t f16_sign = sign << 15;
    uint16_t f16_exp, f16_frac;
    
    // Handle special cases
    if (exp == 0xFF) { 
        f16_exp = 0x1F;
        f16_frac = (frac != 0) ? 0x200 : 0;  
    } else if (exp == 0) { 
        f16_exp = 0;
        f16_frac = 0;
    } else {
        int32_t new_exp = (int32_t)exp - 127 + 15; 
        
        if (new_exp >= 0x1F) { 
            f16_exp = 0x1F;
            f16_frac = 0;
        } else if (new_exp <= 0) {  
            f16_exp = 0;
            f16_frac = 0;
        } else {
            f16_exp = new_exp;
            f16_frac = frac >> 13; 
        }
    }
    
    return f16_sign | (f16_exp << 10) | f16_frac;
}

u8  send_constant_finish=0;
u8  forward_w_init_finish=0;
u8  forward_b_init_finish=0;
u8  forward_x_init_finish=0;

static volatile int cdma_done = 0;
static volatile int cdma_error = 0;
void cdma_isr(void *CallbackRef) {
    u32 sr = Xil_In32(CDMA_SR);

    if (sr & SR_IOC_IRQ) {
        Xil_Out32(CDMA_SR, SR_IOC_IRQ);
        cdma_done = 1;
    }

    if (sr & SR_ERR_IRQ) {
        Xil_Out32(CDMA_SR, SR_ERR_IRQ);
        cdma_error = 1;
        xil_printf("[ISR] CDMA Error! SR=0x%08x\r\n", sr);
    }
}

static int cdma_copy_chunk(uintptr_t sa, uintptr_t da, uint32_t len) {
    u32 timeout_counter = 0;
    const u32 TIMEOUT_MAX = 1000000;
    u32 sr;

    /* Wait for CDMA to be idle before starting new transfer */
    timeout_counter = 0;
    while (!(Xil_In32(CDMA_SR) & SR_IDLE)) {
        if (timeout_counter++ > TIMEOUT_MAX) {
            xil_printf("[Error] CDMA not idle before transfer, timeout!\r\n");
            return -1;
        }
    }

    /* Start CDMA transfer (polling mode - no interrupts) */
    Xil_Out32(CDMA_SA, sa);
    Xil_Out32(CDMA_DA, da);
    Xil_Out32(CDMA_BTT, len);

    /* Poll status register until transfer complete */
    timeout_counter = 0;
    while (1) {
        sr = Xil_In32(CDMA_SR);
        
        /* Check for errors */
        if (sr & SR_ERR_IRQ) {
            xil_printf("[Error] CDMA transfer error! SR=0x%08x\r\n", sr);
            Xil_Out32(CDMA_SR, SR_ERR_IRQ);  // Clear error flag
            return -1;
        }
        
        /* Check if transfer complete (IDLE bit set means done) */
        if (sr & SR_IDLE) {
            /* Clear completion flag if set */
            if (sr & SR_IOC_IRQ) {
                Xil_Out32(CDMA_SR, SR_IOC_IRQ);
            }
            break;  // Transfer complete
        }
        
        /* Timeout check */
        if (timeout_counter++ > TIMEOUT_MAX) {
            xil_printf("[Error] CDMA transfer timeout! SR=0x%08x\r\n", sr);
            return -1;
        }
    }
    return 0;
}
void send_constant_func()
{
	float H_param;
	float E_param;
	float C_param;
	float C;
	int block_size;
	
	H_param = 1.0f/H;

	block_size = hidden / N;  // 256 / 8 = 32
	// C = 1 / sqrt(2 * ln(block_size))
	C = 0.3467f;  
	C_param = C/(2.0f*block_size);
	
	E_param = E;
	
	fp H_bits = fp32_to_fp16(H_param);
	fp C_bits = fp32_to_fp16(C_param);
	fp E_bits = fp32_to_fp16(E_param);

	Xil_Out32(RLN_BASE+reg2, (uint32_t)H_bits);
	Xil_Out32(RLN_BASE+reg3, (uint32_t)C_bits);
	Xil_Out32(RLN_BASE+reg4, (uint32_t)E_bits);
	
	send_constant_finish = 1;
}

int forward_w_init_func()
{
	uintptr_t sa = (uintptr_t)WEIGHT_BASE_ADDR;
	uintptr_t da = (uintptr_t)RLN_BASE_AXIF;
	uintptr_t sa_ps = sa + 0x500000000; 

	if(PL_feedback==0x00000002)
	{
		if (cdma_copy_chunk(sa, da, total_B)) {
			xil_printf("[Error] Weight init CDMA transfer failed!\r\n");
			return -1;
		}

		forward_w_init_finish = 1;
	}
	return 0;
}
int forward_b_init_func()
{
	uintptr_t sa = (uintptr_t)BIAS_BASE_ADDR;
	uintptr_t da = (uintptr_t)RLN_BASE_AXIF;
	uintptr_t sa_ps = sa + 0x500000000;  
	if(PL_feedback==0x00000004)
	{
		if (cdma_copy_chunk(sa_ps, da, total_B)) {
			xil_printf("[Error] bias init CDMA transfer failed!\r\n");
			return -1;
		}
		forward_b_init_finish = 1;
	}
	return 0;
}

int forward_x_init_func()
{
	uintptr_t sa = (uintptr_t)X_BASE_ADDR;
	uintptr_t da = (uintptr_t)RLN_BASE_AXIF;
	uintptr_t sa_ps = sa + 0x500000000;  
	if(PL_feedback==0x00000008)
	{
		if (cdma_copy_chunk(sa, da, total_B)) {
			xil_printf("[Error] X init CDMA transfer failed!\r\n");
			return -1;
		}
		forward_x_init_finish = 1;
	}
	return 0;
}

u8  forward_send_norm_finish=0;
u8  forward_send_out_finish=0;
int forward_send_norm_func()
{
	uintptr_t sa = (uintptr_t)RLN_BASE_AXIF;
	uintptr_t da = (uintptr_t)NORM_ADDR;
	uintptr_t da_ps = da + 0x500000000;
	if(PL_feedback==0x00000010)
	{
		u32 fm_cycles = Xil_In32(RLN_BASE + reg6);  // status_reg1 = FM_cal_clk
		xil_printf("  FM Calculation Cycles: %u\r\n", fm_cycles);

		uintptr_t clk_addr = (uintptr_t)CAL_CLK_ADDR + PS_DDR_OFFSET;
		Xil_Out32(clk_addr, fm_cycles);

		if (cdma_copy_chunk(sa, da, total_B)) {
			xil_printf("[Error] Norm send CDMA transfer failed!\r\n");
			return -1;
		}
		
		forward_send_norm_finish = 1;
	}
	return 0;
}


int forward_send_out_func()
{
	uintptr_t sa = (uintptr_t)RLN_BASE_AXIF;
	uintptr_t da = (uintptr_t)FW_RESULT_ADDR;
	uintptr_t da_ps = da + 0x500000000;
	if(PL_feedback==0x00000020)
	{
		if (cdma_copy_chunk(sa, da, total_B)) {
			xil_printf("[Error] FM_out send CDMA transfer failed!\r\n");
			return -1;
		}
		
		forward_send_out_finish = 1;
	}
	return 0;
}

u8  backward_param_y_init_finish=0;
u8  backward_param_dz_init_finish=0;
int backward_param_y_init_func()
{

	uintptr_t seq0_addr = (uintptr_t)NORM_ADDR + PS_DDR_OFFSET;
	uintptr_t seq1_addr = seq0_addr + SEQ_OFFSET;
	uintptr_t seq2_addr = seq1_addr + SEQ_OFFSET;
	uintptr_t seq3_addr = seq2_addr + SEQ_OFFSET;
	u16 seq0_fp16, seq1_fp16, seq2_fp16, seq3_fp16;
	u32 y_data_reg2; 
	u32 y_data_reg3;  
	
	if(PL_feedback==0x00000080)
	{
		for(int i = 0; i < hidden; i++)
		{
			seq0_fp16 = Xil_In16(seq0_addr + i*2);
			seq1_fp16 = Xil_In16(seq1_addr + i*2);
			
			seq2_fp16 = Xil_In16(seq2_addr + i*2);
			seq3_fp16 = Xil_In16(seq3_addr + i*2);
			
			y_data_reg2 = ((u32)seq1_fp16 << 16) | (u32)seq0_fp16;

			y_data_reg3 = ((u32)seq3_fp16 << 16) | (u32)seq2_fp16;
			
			Xil_Out32(RLN_BASE + reg4, i);

			Xil_Out32(RLN_BASE + reg2, y_data_reg2);
			
			Xil_Out32(RLN_BASE + reg3, y_data_reg3);
			
		}
		
		backward_param_y_init_finish = 1;
		return 0;
	}
	
	return -1;
};

int backward_param_dz_init_func()
{
	uintptr_t seq0_addr = (uintptr_t)DELTA_BASE_ADDR + PS_DDR_OFFSET;
	uintptr_t seq1_addr = seq0_addr + SEQ_OFFSET;
	uintptr_t seq2_addr = seq1_addr + SEQ_OFFSET;
	uintptr_t seq3_addr = seq2_addr + SEQ_OFFSET;
	u16 seq0_fp16, seq1_fp16, seq2_fp16, seq3_fp16;
	u32 dz_data_reg2;  
	u32 dz_data_reg3;  
	
	if(PL_feedback==0x00000100)
	{
		for(int i = 0; i < hidden; i++)
		{
			seq0_fp16 = Xil_In16(seq0_addr + i*2);
			seq1_fp16 = Xil_In16(seq1_addr + i*2);
			
			seq2_fp16 = Xil_In16(seq2_addr + i*2);
			seq3_fp16 = Xil_In16(seq3_addr + i*2);
			
			dz_data_reg2 = ((u32)seq1_fp16 << 16) | (u32)seq0_fp16;

			dz_data_reg3 = ((u32)seq3_fp16 << 16) | (u32)seq2_fp16;

			Xil_Out32(RLN_BASE + reg4, i);

			Xil_Out32(RLN_BASE + reg2, dz_data_reg2);

			Xil_Out32(RLN_BASE + reg3, dz_data_reg3);
		}
		
		backward_param_dz_init_finish = 1;
		return 0;
	}
	
	return -1;
};

u8  backward_param_send_result_finish=0;
int backward_param_send_result_func()
{
	uintptr_t gradg_addr = (uintptr_t)BW_W_RESULT_ADDR + PS_DDR_OFFSET;
	uintptr_t gradb_addr = (uintptr_t)BW_B_RESULT_ADDR + PS_DDR_OFFSET;
	u32 bm_param_cycles;
	u32 result;
	u16 gradb_fp16;
	u16 gradg_fp16;
	
	if(PL_feedback==0x00000200)
	{
		bm_param_cycles = Xil_In32(RLN_BASE + reg6);  // status_reg1 = BM_Param_cal_clk
		xil_printf("  BM Param Calculation Cycles: %u\r\n", bm_param_cycles);
		
		uintptr_t clk_addr = (uintptr_t)CAL_CLK_ADDR + PS_DDR_OFFSET + 4;
		Xil_Out32(clk_addr, bm_param_cycles);
		xil_printf("  BM Param Cycles saved to DDR: 0x%llx\r\n", clk_addr);

		for(int i = 0; i < hidden; i++)
		{
			Xil_Out32(RLN_BASE + reg4, i);
			
			result = Xil_In32(RLN_BASE + reg7);
			
			gradb_fp16 = result & 0xFFFF;        
			gradg_fp16 = (result >> 16) & 0xFFFF; 
			
			Xil_Out16(gradg_addr + i*2, gradg_fp16);
			Xil_Out16(gradb_addr + i*2, gradb_fp16);
		}
		backward_send_result_finish = 1;
		return 0;
	}
	
	return -1;
}


u8 backward_dz_init_finish=0;
u8 backward_send_result_finish=0;
int backward_dz_init_func()
{
	uintptr_t sa = (uintptr_t)DELTA_BASE_ADDR;
	uintptr_t da = (uintptr_t)RLN_BASE_AXIF;
	uintptr_t sa_ps = sa + 0x500000000;  
	if(PL_feedback==0x00000400)
	{
		if (cdma_copy_chunk(sa, da, total_B)) {
			xil_printf("[Error] DZ init CDMA transfer failed!\r\n");
			return -1;
		}
		backward_dz_init_finish = 1;
	}
	return 0;
}

int backward_send_result_func()
{
	uintptr_t sa = (uintptr_t)RLN_BASE_AXIF;
	uintptr_t da = (uintptr_t)BW_X_RESULT_ADDR;
	uintptr_t da_ps = da + 0x500000000;
	if(PL_feedback==0x00000800)
	{
		u32 bm_cycles = Xil_In32(RLN_BASE + reg6);  // status_reg1 = BM_cal_clk
		xil_printf("  BM Calculation Cycles: %u\r\n", bm_cycles);

		uintptr_t clk_addr = (uintptr_t)CAL_CLK_ADDR + PS_DDR_OFFSET + 8;
		Xil_Out32(clk_addr, bm_cycles);

		if (cdma_copy_chunk(sa, da, total_B)) {
			xil_printf("[Error] BM gradx send CDMA transfer failed!\r\n");
			return -1;
		}
	
		backward_send_result_finish = 1;
	}
	return 0;
}

