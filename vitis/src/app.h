#ifndef __APP_
#define __APP_

#include "define.h"
#include "xil_printf.h"
#include "xparameters.h"

enum main_state_top
{
	idle_main,idle_send_constant,send_constant,idle_forward,forward_w_init,forward_b_init,
	forward_x_init,forward_work,forward_send_norm,forward_send_out,
	idle_BM_Param,BM_Param_initial_y,BM_Param_initial_dz,BM_Param_work,BM_Param_send_result,
	idle_BM,BM_initial_dz,BM_work,BM_send_result
};


void app_main(void);

#endif
