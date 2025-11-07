#ifndef __PL_DATA_MANAGE_
#define __PL_DATA_MANAGE_

#include <xbasic_types.h>

extern u8 send_constant_finish;
extern u8 forward_w_init_finish;
extern u8 forward_b_init_finish;
extern u8 forward_x_init_finish;
extern u8 forward_send_norm_finish;
extern u8 forward_send_out_finish;
extern u8 backward_param_y_init_finish;
extern u8 backward_param_dz_init_finish;
extern u8 backward_param_send_result_finish;
extern u8 backward_dz_init_finish;
extern u8 backward_send_result_finish;

void send_constant_func(void);
int forward_w_init_func(void);
int forward_b_init_func(void);
int forward_x_init_func(void);
int forward_send_norm_func(void);
int forward_send_out_func(void);
int backward_param_y_init_func(void);
int backward_param_dz_init_func(void);
int backward_param_send_result_func(void);
int backward_dz_init_func(void);
int backward_send_result_func(void);

void cdma_isr(void *CallbackRef);


#endif
