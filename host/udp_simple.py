#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单UDP示例 - 激活板子PS程序并等待完成信号
"""

import socket
import struct
import time

# 配置
BOARD_IP = "192.168.1.10"
BOARD_PORT = 5001
LOCAL_IP = "192.168.1.20"
LOCAL_PORT = 8080
COMPLETION_SIGNAL = 0x0001  # 板子完成信号

# 创建socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))
# 不设置超时，让recv一直阻塞等待
print(f"监听地址: {LOCAL_IP}:{LOCAL_PORT}")

# 发送激活命令 0x00000001
activate_cmd = struct.pack('<I', 0x00000001)
sock.sendto(activate_cmd, (BOARD_IP, BOARD_PORT))
print(f"✓ 已发送激活命令 0x00000001 到 {BOARD_IP}:{BOARD_PORT}")
print(f"等待板子完成信号 0x{COMPLETION_SIGNAL:04X}...\n")

# 一直阻塞等待完成信号
start_time = time.time()
while True:
    try:
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
                print(f"总耗时: {elapsed:.2f} 秒")
                print(f"{'='*60}\n")
                break
            else:
                print(f"  ⚠ 非完成信号，继续等待...")
        else:
            print(f"  ⚠ 数据长度不足，继续等待...")
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
        break
    except Exception as e:
        print(f"❌ 接收错误: {e}")
        break

sock.close()
print("程序结束\n")

