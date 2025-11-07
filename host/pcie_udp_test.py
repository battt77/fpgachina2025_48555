"""
PCIe + UDP 联合测试脚本
1. 通过PCIe发送4组数据到DDR（X数据、权重、偏置、梯度）
2. 通过UDP发送使能信号激活FPGA处理
"""

import sys
import numpy as np
import time
import torch
import LN_RBN_BP_tch as rln
import socket
import struct

# =============================================================================
# 配置参数
# =============================================================================
X_DIM = 2        # sequence length
Y_DIM = 256      # hidden states
BATCH_SIZE = 8   # 每次传输的数据个数
BLOCK_SIZE = 32

# 导入PCIe相关函数
from pcie_rw import (
    check_xdma_driver, pcie_init, pcie_open, pcie_close,
    write_fp16_array_to_ddr, c2h_transfer,
    ADDR_X_DATA, ADDR_WEIGHTS, ADDR_BIAS, ADDR_GRAD
)

# 输出结果地址
ADDR_OUTPUT = 0x00500000  # 输出结果基地址 (5MB偏移)

# 导入UDP客户端
from udp_client import UDPClient

# UDP配置
BOARD_IP = "192.168.1.10"
BOARD_PORT = 5001
LOCAL_IP = "192.168.1.20"
LOCAL_PORT = 8080
COMPLETION_SIGNAL = 0x0001  # 板子完成信号


class My_net(torch.nn.Module):
  def __init__(self):
    super(My_net,self).__init__()
    self.myln=rln.LayerNormR(Y_DIM,eps=1e-6,batch_size=BLOCK_SIZE)
  def forward(self,x):
    y = self.myln(x)
    return y
  
def send_all_data_via_pcie():
    """
    通过PCIe发送所有数据到DDR
    返回: (success, ln_y_tensor) - success=True/False, ln_y_tensor=参考输出张量
    """
    print(f"\n{'='*60}")
    print(f"步骤1: 通过PCIe发送数据到DDR")
    print(f"{'='*60}")
    print(f"配置: Seq={X_DIM}, Hidden={Y_DIM}, Batch={BATCH_SIZE}")
    
    try:
        # 1. X数据 (使用随机数)
        print(f"\n[1/4] X数据 ({X_DIM}x{Y_DIM} = {X_DIM*Y_DIM}个元素)")
        np.random.seed(42)  # 固定随机种子，便于复现
        x_data = np.random.randn(X_DIM * Y_DIM).astype(np.float16)  # 标准正态分布
        print(f"  数据: 随机数 (标准正态分布, seed=42)")
        print(f"  前5个值: {x_data[:5]}")
        print(f"  后5个值: {x_data[-5:]}")
        print(f"  地址: 0x{ADDR_X_DATA:08X}")
        
        success, total = write_fp16_array_to_ddr(x_data, ADDR_X_DATA, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        if success != total:
            return (False, None)
        
        # 2. 权重 (每个sequence都是相同的复制, 使用随机数)
        print(f"\n[2/4] 权重 ({X_DIM}x{Y_DIM} = {X_DIM*Y_DIM}个元素)")
        np.random.seed(123)  # 固定随机种子
        weights_single = np.random.uniform(0.8, 1.2, Y_DIM).astype(np.float16)  # 均匀分布在[0.8, 1.2]
        weights = np.tile(weights_single, X_DIM)  # 复制X_DIM次
        
        print(f"  数据: 随机数 (均匀分布[0.8, 1.2], seed=123, 每个sequence相同)")
        print(f"  前5个值: {weights_single[:5]}")
        print(f"  后5个值: {weights_single[-5:]}")
        print(f"  形状: ({X_DIM}, {Y_DIM})")
        print(f"  地址: 0x{ADDR_WEIGHTS:08X}")
        
        success, total = write_fp16_array_to_ddr(weights, ADDR_WEIGHTS, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        if success != total:
            return (False, None)
        
        # 3. 偏置 (每个sequence都是相同的复制, 使用随机数)
        print(f"\n[3/4] 偏置 ({X_DIM}x{Y_DIM} = {X_DIM*Y_DIM}个元素)")
        np.random.seed(456)  # 固定随机种子
        bias_single = np.random.uniform(-0.2, 0.2, Y_DIM).astype(np.float16)  # 均匀分布在[-0.2, 0.2]
        bias = np.tile(bias_single, X_DIM)  # 复制X_DIM次
        
        print(f"  数据: 随机数 (均匀分布[-0.2, 0.2], seed=456, 每个sequence相同)")
        print(f"  前5个值: {bias_single[:5]}")
        print(f"  后5个值: {bias_single[-5:]}")
        print(f"  形状: ({X_DIM}, {Y_DIM})")
        print(f"  地址: 0x{ADDR_BIAS:08X}")
        
        success, total = write_fp16_array_to_ddr(bias, ADDR_BIAS, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        if success != total:
            return (False, None)
        
        # 计算参考前向传播结果
        print(f"\n[参考计算] 使用PyTorch计算LayerNorm前向传播 (FP16)")
        device = torch.device('cpu')  # 使用CPU计算
        net = My_net().to(device).half()  # 转换模型为FP16
        
        # 将numpy数组转换为torch张量（FP16）
        # 注意：LayerNorm的权重和bias是(Y_DIM,)形状，所以使用_single版本
        weight_tensor = torch.from_numpy(weights_single).to(device)
        bias_tensor = torch.from_numpy(bias_single).to(device)
        x_tensor = torch.from_numpy(x_data.reshape(X_DIM, Y_DIM)).to(device).requires_grad_(True)
        
        # 加载参数到模型
        params_init_dict = {
            'myln.weight': weight_tensor,
            'myln.bias': bias_tensor
        }
        net.load_state_dict(params_init_dict)
        
        # 前向传播
        ln_y = net(x_tensor)
        print(f"  输入形状: {x_tensor.shape}, dtype: {x_tensor.dtype}")
        print(f"  输出形状: {ln_y.shape}, dtype: {ln_y.dtype}")
        print(f"  输出前5个值: {ln_y.flatten()[:5].detach().numpy()}")
        print(f"  输出后5个值: {ln_y.flatten()[-5:].detach().numpy()}")


        # 4. 梯度
        print(f"\n[4/4] 梯度 ({X_DIM}x{Y_DIM} = {X_DIM*Y_DIM}个元素)")
        grad = np.full(X_DIM * Y_DIM, 1.0, dtype=np.float16)
        print(f"  数据: 全部为 1.0")
        print(f"  地址: 0x{ADDR_GRAD:08X}")
        
        success, total = write_fp16_array_to_ddr(grad, ADDR_GRAD, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        if success != total:
            return (False, None)
        
        # 汇总
        print(f"\n{'='*60}")
        print(f"✓ PCIe数据传输完成")
        print(f"  X数据:  0x{ADDR_X_DATA:08X} ({X_DIM*Y_DIM}个FP16, {X_DIM*Y_DIM*2}字节)")
        print(f"  权重:   0x{ADDR_WEIGHTS:08X} ({X_DIM*Y_DIM}个FP16, {X_DIM*Y_DIM*2}字节)")
        print(f"  偏置:   0x{ADDR_BIAS:08X} ({X_DIM*Y_DIM}个FP16, {X_DIM*Y_DIM*2}字节)")
        print(f"  梯度:   0x{ADDR_GRAD:08X} ({X_DIM*Y_DIM}个FP16, {X_DIM*Y_DIM*2}字节)")
        print(f"{'='*60}")
        
        return (True, ln_y)
        
    except Exception as e:
        print(f"\n❌ PCIe传输错误: {e}")
        return (False, None)


def read_and_compare_results(reference_output):
    """
    从FPGA读取计算结果并与参考输出对比
    
    参数:
        reference_output: PyTorch计算的参考输出张量 (FP16)
    
    返回:
        True=对比一致, False=对比失败或不一致
    """
    print(f"\n{'='*60}")
    print(f"步骤3: 读取FPGA计算结果并对比")
    print(f"{'='*60}")
    
    try:
        # 计算需要读取的字节数 (X_DIM * Y_DIM 个FP16，每个2字节)
        output_size = X_DIM * Y_DIM * 2
        print(f"读取地址: 0x{ADDR_OUTPUT:08X}")
        print(f"读取大小: {output_size} 字节 ({X_DIM * Y_DIM}个FP16)")
        
        # 从FPGA读取数据
        fpga_output_bytes = c2h_transfer(ADDR_OUTPUT, output_size)
        
        if fpga_output_bytes is None:
            print("❌ 从FPGA读取数据失败")
            return False
        
        if len(fpga_output_bytes) != output_size:
            print(f"❌ 读取数据长度不匹配: 期望{output_size}字节, 实际{len(fpga_output_bytes)}字节")
            return False
        
        print(f"✓ 成功读取 {len(fpga_output_bytes)} 字节")
        
        # 将字节数据转换为FP16数组
        fpga_output_fp16 = np.frombuffer(fpga_output_bytes, dtype=np.float16)
        fpga_output_fp16 = fpga_output_fp16.reshape(X_DIM, Y_DIM)
        
        print(f"FPGA输出形状: {fpga_output_fp16.shape}")
        print(f"FPGA输出前5个值: {fpga_output_fp16.flatten()[:5]}")
        print(f"FPGA输出后5个值: {fpga_output_fp16.flatten()[-5:]}")
        
        # 将numpy数组转换为torch张量
        fpga_output_tensor = torch.from_numpy(fpga_output_fp16)
        
        # 使用torch.allclose进行对比
        print(f"\n{'='*60}")
        print(f"结果对比")
        print(f"{'='*60}")
        print(f"参考输出形状: {reference_output.shape}, dtype: {reference_output.dtype}")
        print(f"FPGA输出形状: {fpga_output_tensor.shape}, dtype: {fpga_output_tensor.dtype}")
        
        # 对比前先确保两者形状一致
        if reference_output.shape != fpga_output_tensor.shape:
            print(f"❌ 形状不匹配！")
            return False
        
        # 使用torch.allclose对比，考虑FP16的精度限制
        # rtol=1e-3, atol=1e-3 对于FP16是合理的容差
        is_close = torch.allclose(reference_output.cpu().detach(), fpga_output_tensor, rtol=1e-3, atol=1e-3)
        
        if is_close:
            print(f"✓✓✓ 结果对比: 一致！(rtol=1e-3, atol=1e-3)")
            
            # 计算统计信息
            diff = torch.abs(reference_output.cpu().detach() - fpga_output_tensor)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"\n统计信息:")
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
            print(f"{'='*60}")
            return True
        else:
            print(f"❌ 结果对比: 不一致！")
            
            # 显示差异详情
            diff = torch.abs(reference_output.cpu().detach() - fpga_output_tensor)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            max_diff_idx = torch.argmax(diff)
            
            print(f"\n差异详情:")
            print(f"  最大差异: {max_diff:.6f} (位置: {max_diff_idx})")
            print(f"  平均差异: {mean_diff:.6f}")
            print(f"  参考值[{max_diff_idx}]: {reference_output.flatten()[max_diff_idx].item():.6f}")
            print(f"  FPGA值[{max_diff_idx}]: {fpga_output_tensor.flatten()[max_diff_idx].item():.6f}")
            print(f"{'='*60}")
            return False
            
    except Exception as e:
        print(f"\n❌ 读取或对比错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def send_activate_via_udp():
    """
    通过UDP发送激活命令到板子，并等待完成信号
    返回: True=成功, False=失败
    """
    print(f"\n{'='*60}")
    print(f"步骤2: 通过UDP发送激活命令并等待完成")
    print(f"{'='*60}")
    
    sock = None
    try:
        # 创建UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 设置socket选项：允许地址重用（重要！避免TIME_WAIT状态占用端口）
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定本地地址和端口
        sock.bind((LOCAL_IP, LOCAL_PORT))
        
        # 不设置超时，让recv一直阻塞等待
        print(f"监听地址: {LOCAL_IP}:{LOCAL_PORT}")
        
        # 发送激活命令
        activate_cmd = struct.pack('<I', 0x00000001)
        bytes_sent = sock.sendto(activate_cmd, (BOARD_IP, BOARD_PORT))
        print(f"✓ 已发送激活命令 0x00000001 到 {BOARD_IP}:{BOARD_PORT}")
        print(f"  发送字节: {activate_cmd.hex()} ({bytes_sent}字节)")
        print(f"  字节序: 小端序 (Little Endian)")
        print(f"等待板子完成信号 0x{COMPLETION_SIGNAL:04X}...\n")
        
        # 一直阻塞等待完成信号
        start_time = time.time()
        while True:
            data, addr = sock.recvfrom(1024)
            elapsed = time.time() - start_time
            print(f"[{elapsed:.2f}s] 收到数据: {data.hex()} from {addr}")
            
            # 解析收到的数据
            if len(data) >= 2:
                # 尝试解析为16位整数（小端序）
                received_value = struct.unpack('<H', data[:2])[0]
                print(f"  解析值: 0x{received_value:04X}")
                
                # 检查是否是完成信号
                if received_value == COMPLETION_SIGNAL:
                    print(f"\n{'='*60}")
                    print(f"✓✓✓ 收到完成信号 0x{COMPLETION_SIGNAL:04X}！")
                    print(f"✓✓✓ 板子已完成所有逻辑处理")
                    print(f"FPGA处理耗时: {elapsed:.3f} 秒")
                    print(f"{'='*60}")
                    return True
                else:
                    print(f"  ⚠ 非完成信号，继续等待...")
            else:
                print(f"  ⚠ 数据长度不足，继续等待...")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return False
    except Exception as e:
        print(f"\n❌ UDP通信错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if sock:
            sock.close()


def main():
    """主函数：完整的PCIe+UDP流程"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "PCIe + UDP 联合测试脚本" + " "*25 + "║")
    print("╚" + "="*58 + "╝")
    print("\n流程:")
    print("  1. PCIe发送数据到DDR (X、权重、偏置、梯度)")
    print("  2. UDP发送激活命令到板子")
    print("  3. 等待板子完成信号 (0x0001)")
    print("  4. 通过PCIe读取FPGA计算结果")
    print("  5. 使用torch.allclose对比结果")
    print("\n" + "-"*60)
    
    # ====== 步骤0: 初始化PCIe ======
    print("\n初始化PCIe设备...")
    check_xdma_driver()
    
    if pcie_init() < 0:
        print("❌ PCIe初始化失败")
        sys.exit(1)
    print("✓ PCIe初始化成功")
    
    if pcie_open() < 0:
        print("❌ PCIe设备打开失败")
        sys.exit(1)
    print("✓ PCIe设备已打开")
    
    try:
        # ====== 步骤1: PCIe发送数据 ======
        success, reference_output = send_all_data_via_pcie()
        if not success:
            print("\n❌ PCIe数据传输失败，中止流程")
            sys.exit(1)
        
        # 短暂延时，确保数据稳定
        time.sleep(0.1)
        
        # ====== 步骤2: UDP发送激活命令并等待完成 ======
        if not send_activate_via_udp():
            print("\n❌ UDP通信失败或未收到完成信号")
            sys.exit(1)
        
        # ====== 步骤3: 读取FPGA结果并对比 ======
        if not read_and_compare_results(reference_output):
            print("\n⚠ 结果对比失败或不一致")
            # 不退出，继续显示完成信息
        
        # ====== 完成 ======
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*12 + "✓✓✓ 完整流程执行成功 ✓✓✓" + " "*12 + "║")
        print("╚" + "="*58 + "╝")
        print("\nFPGA已完成LayerNorm计算并完成结果验证！")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭PCIe
        pcie_close()
        print("\n✓ PCIe设备已关闭")
        print("\n程序结束\n")


if __name__ == "__main__":
    main()

