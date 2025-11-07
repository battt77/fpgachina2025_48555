#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP客户端 - 与FPGA板子通信
板子: 192.168.1.10:5001
本机: 192.168.1.20:8080
"""

import socket
import struct
import time

# 配置参数
BOARD_IP = "192.168.1.10"
BOARD_PORT = 5001
LOCAL_IP = "192.168.1.20"
LOCAL_PORT = 8080

# 激活命令
CMD_ACTIVATE = 0x00000001


class UDPClient:
    def __init__(self, board_ip=None, board_port=None, local_ip=None, local_port=None):
        """
        初始化UDP客户端
        board_ip: 板子IP，默认使用BOARD_IP
        board_port: 板子端口，默认使用BOARD_PORT
        local_ip: 本机IP，默认使用LOCAL_IP
        local_port: 本机端口，默认使用LOCAL_PORT
        """
        self.sock = None
        self.board_ip = board_ip or BOARD_IP
        self.board_port = board_port or BOARD_PORT
        self.local_ip = local_ip or LOCAL_IP
        self.local_port = local_port or LOCAL_PORT
        self.board_addr = (self.board_ip, self.board_port)
        
    def init(self):
        """初始化UDP socket"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.local_ip, self.local_port))
            self.sock.settimeout(2.0)  # 2秒超时
            print(f"✓ UDP客户端初始化成功")
            print(f"  本机: {self.local_ip}:{self.local_port}")
            print(f"  目标: {self.board_ip}:{self.board_port}")
            return True
        except Exception as e:
            print(f"❌ UDP初始化失败: {e}")
            return False
    
    def send_activate(self):
        """发送激活命令"""
        try:
            # 打包为4字节小端序
            data = struct.pack('<I', CMD_ACTIVATE)
            self.sock.sendto(data, self.board_addr)
            print(f"✓ 发送激活命令: 0x{CMD_ACTIVATE:08X}")
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def send_data(self, data):
        """
        发送数据到板子
        data: bytes 或 int
        """
        try:
            if isinstance(data, int):
                data = struct.pack('<I', data)
            elif isinstance(data, (list, tuple)):
                data = struct.pack(f'<{len(data)}I', *data)
            
            self.sock.sendto(data, self.board_addr)
            print(f"✓ 发送数据: {len(data)}字节")
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def receive(self, buffer_size=1024):
        """接收数据"""
        try:
            data, addr = self.sock.recvfrom(buffer_size)
            print(f"✓ 接收到数据: {len(data)}字节 from {addr}")
            return data
        except socket.timeout:
            print(f"⚠ 接收超时")
            return None
        except Exception as e:
            print(f"❌ 接收失败: {e}")
            return None
    
    def send_and_receive(self, data, buffer_size=1024):
        """发送数据并等待响应"""
        if self.send_data(data):
            return self.receive(buffer_size)
        return None
    
    def close(self):
        """关闭socket"""
        if self.sock:
            self.sock.close()
            print(f"✓ UDP连接已关闭")


def test_activate():
    """测试：发送激活命令"""
    client = UDPClient()
    
    if not client.init():
        return
    
    try:
        print("\n" + "="*50)
        print("发送激活命令到板子")
        print("="*50)
        
        # 发送激活命令
        if client.send_activate():
            print("等待板子响应...")
            response = client.receive()
            if response:
                print(f"收到响应: {response.hex()}")
            else:
                print("未收到响应（板子可能已激活）")
        
    finally:
        client.close()


def test_continuous():
    """测试：持续发送数据"""
    client = UDPClient()
    
    if not client.init():
        return
    
    try:
        print("\n" + "="*50)
        print("持续发送测试")
        print("="*50)
        
        for i in range(10):
            print(f"\n第 {i+1} 次:")
            client.send_data(CMD_ACTIVATE)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        client.close()


def send_fp16_data():
    """发送FP16数据到板子"""
    import numpy as np
    
    client = UDPClient()
    if not client.init():
        return
    
    try:
        print("\n" + "="*50)
        print("发送FP16数据")
        print("="*50)
        
        # 准备8个FP16数据
        data_fp32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        data_fp16 = data_fp32.astype(np.float16)
        data_bytes = data_fp16.tobytes()
        
        print(f"数据: {data_fp32}")
        print(f"FP16: {data_bytes.hex()}")
        print(f"长度: {len(data_bytes)}字节")
        
        # 发送数据
        client.sock.sendto(data_bytes, client.board_addr)
        print("✓ 发送成功")
        
        # 等待响应
        response = client.receive()
        if response:
            print(f"响应: {response.hex()}")
        
    finally:
        client.close()


if __name__ == "__main__":
    import sys
    
    print("="*50)
    print("UDP客户端 - FPGA板子通信")
    print("="*50)
    print("1. 发送激活命令")
    print("2. 持续发送测试")
    print("3. 发送FP16数据")
    print("="*50)
    
    choice = input("请选择 (1/2/3) [默认=1]: ").strip()
    if not choice:
        choice = "1"
    
    if choice == "1":
        test_activate()
    elif choice == "2":
        test_continuous()
    elif choice == "3":
        send_fp16_data()
    else:
        print("无效选择")

