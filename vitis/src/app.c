#include "app.h"
#include "xil_io.h"
#include "PL_data_manage.h"
#include "udp_func.h"

enum  main_state_top main_state;

void PL_control_get(void);

u32 PL_control=0;
u32 PL_feedback=0;
u32 PL_enable=0;


u8 idle_wait_recv=1;


void app_fsm_state()
{
	switch(main_state)
	{
		case(idle_main):
//			xil_printf("idle!\r\n");
			if(app_run)
			{
				main_state=idle_send_constant;
				xil_printf("send_constant_idle\r\n");
				app_run=0;
				PL_enable=0x00000001;
			}
		break;
		case(idle_send_constant):
			if(PL_feedback==0x00000001)
			{
				xil_printf("PL_feedback:0x%x\r\n",PL_feedback);
				PL_enable=0;
				main_state=send_constant;
				xil_printf("send_constant_on!\r\n");
				//PL_enable=0x00000002;
			}
		break;
		case(send_constant):
			if(send_constant_finish)
			{
				send_constant_finish=0;
				main_state=idle_forward;
				PL_control=0x00000002;
				xil_printf("idle_forward!\r\n");
			}
		break;
		case(idle_forward):
			if(PL_feedback==0x00000002)
			{
				main_state=forward_w_init;
				PL_control=0;
				xil_printf("forward_w_init!\r\n");
				PL_enable=0;
			}
		break;
		case(forward_w_init):
			if(forward_w_init_finish)
			{
				forward_w_init_finish=0;
				main_state=forward_b_init;
				PL_control=0x00000004;
				xil_printf("forward_b_init!\r\n");
			}
		break;
		case(forward_b_init):
			if(forward_b_init_finish)
			{
				forward_b_init_finish=0;
				main_state=forward_x_init;
				PL_control=0x00000008;
				xil_printf("forward_x_init!\r\n");
			}
		break;
		case(forward_x_init):
			if(forward_x_init_finish)
			{
				forward_x_init_finish=0;
				main_state=forward_work;
				PL_control=0x00000010;
				xil_printf("forward_work!\r\n");
			}
		break;
		case(forward_work):
			if(PL_feedback==0x00000010)
			{
				main_state=forward_send_norm;
				PL_control=0;
				xil_printf("forward_send_norm!\r\n");
			}
		break;
		case(forward_send_norm):
			if(forward_send_norm_finish)
			{
				forward_send_norm_finish=0;
				main_state=forward_send_out;
				PL_control=0x00000020;
				xil_printf("forward_send_out!\r\n");
			}
		break;
		case(forward_send_out):
			if(forward_send_out_finish)
			{
				forward_send_out_finish = 0;
				main_state=idle_BM_Param;
				PL_control=0x00000040;
				xil_printf("idle_BM_Param!\r\n");
			}
		break;
		case(idle_BM_Param):
			if(PL_feedback==0x00000080)
			{
				main_state=BM_Param_initial_y;
				PL_control=0;
				xil_printf("backward_param_y_init!\r\n");
				PL_enable=0;
			}
		break;
		case(BM_Param_initial_y):
			if(backward_param_y_init_finish)
			{
				backward_param_y_init_finish=0;
				main_state=BM_Param_initial_dz;
				PL_control=0x00000080;
				xil_printf("backward_param_dz_init!\r\n");
			}
		break;
		case(BM_Param_initial_dz):
			if(backward_param_dz_init_finish)
			{
				backward_param_dz_init_finish=0;
				main_state=BM_Param_work;
				PL_control=0x00000100;
				xil_printf("backward_param_work!\r\n");
			}
		break;
		case(BM_Param_work):
			if(PL_feedback==0x00000200)
			{
				main_state=BM_Param_send_result;
				PL_control=0;
				xil_printf("backward_param_result!\r\n");
			}
		break;
		case(BM_Param_send_result):
			if(backward_send_result_finish)
			{
				backward_send_result_finish = 0;
				main_state=idle_BM;
				PL_control=0x00000200;
				xil_printf("idle_BM!\r\n");
			}
		break;
		case(idle_BM):
			if(PL_feedback==0x00000400)
			{
				main_state=BM_initial_dz;
				PL_control=0;
				xil_printf("backward_dz_init!\r\n");
				PL_enable=0;
			}
		break;
		case(BM_initial_dz):
			if(backward_dz_init_finish)
			{
				backward_dz_init_finish=0;
				main_state=BM_work;
				PL_control=0x00000400;
				xil_printf("backward_work!\r\n");
			}
		break;
		case(BM_work):
			if(PL_feedback==0x00000800)
			{
				main_state=BM_send_result;
				PL_control=0;
				xil_printf("backward_send_result!\r\n");
			}
		break;
		case(BM_send_result):
			if(backward_send_result_finish)
			{
				backward_send_result_finish = 0;
				main_state=idle_main;
				PL_control=0x00000800;
				xil_printf("idle_main!\r\n");

				udp_print_finish();
				idle_wait_recv = 1;
			}
		break;
	}
}

void app_fsm_control()
{
	switch(main_state)
	{
		case(idle_main):
			wait_run();
		break;
		case(idle_send_constant):
		break;
		case(send_constant):
			send_constant_func();
		break;
		case(idle_forward):
		   PL_enable=0x00000002;
		break;
		case(forward_w_init):
			forward_w_init_func();
		break;
		case(forward_b_init):
			forward_b_init_func();
		break;
		case(forward_x_init):
			forward_x_init_func();
		break;
		case(forward_work):
		break;
		case(forward_send_norm):
			forward_send_norm_func();
		break;
		case(forward_send_out):
			forward_send_out_func();
		break;
		case(idle_BM_Param):
			PL_enable=0x00000004;
		break;
		case(BM_Param_initial_y):
			backward_param_y_init_func();
		break;
		case(BM_Param_initial_dz):
			backward_param_dz_init_func();
		break;
		case(BM_Param_work):
		break;
		case(BM_Param_send_result):
			backward_param_send_result_func();
		break;
		case(idle_BM):
			PL_enable=0x00000008;
		break;
		case(BM_initial_dz):
			backward_dz_init_func();
		break;
		case(BM_work):
		break;
		case(BM_send_result):
			backward_send_result_func();
		break;
	}
}


void app_main()
{

	app_fsm_state();
	app_fsm_control();
	PL_control_get();
}

void PL_control_get()
{
	u32 tst1;
	//u32 PL_control_swap=0;
	//PL_control_swap=swap_endian32(PL_control);



	Xil_Out32(RLN_BASE+reg0,PL_control); //control
	Xil_Out32(RLN_BASE+reg1,PL_enable); //enable
//	Xil_Out32(RLN_BASE+reg8,0x00040000); //enable
//	Xil_Out32(RLN_BASE+reg5,0x00050000); //enable
	PL_feedback=Xil_In32(RLN_BASE+reg5);
	tst1=Xil_In32(RLN_BASE+reg6);
//	printf("PL_feedback7:0x%x\r\n",tst1);
//	printf("PL_feedback5:0x%x\r\n",PL_feedback);
	/*
	if(main_state==backward_wb_work)
	{
		xil_printf("tst channel2:%x!\r\n",tst1);
	}
*/
}
