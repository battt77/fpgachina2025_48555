"""
Xilinx FPGA XDMA PCIe 读写程序 - Python版本
支持通过XDMA IP进行Host与FPGA之间的数据传输

主要功能：
- pcie_init(): 初始化PCIe设备
- pcie_open(): 打开PCIe设备
- pcie_close(): 关闭PCIe设备
- h2c_transfer(): Host到Card的数据传输（写入FPGA）
- c2h_transfer(): Card到Host的数据传输（从FPGA读取）
"""

# =============================================================================
# Tensor传输参数配置
# =============================================================================
X_DIM = 1        # sequence length
Y_DIM = 256      # hidden states
BATCH_SIZE = 8 # 每次传输的数据个数

# DDR地址映射
ADDR_X_DATA = 0x00000000   # X数据基地址
ADDR_WEIGHTS = 0x00100000  # 权重基地址 (1MB偏移)
ADDR_BIAS = 0x00200000     # 偏置基地址 (2MB偏移)
ADDR_GRAD = 0x00300000     # 梯度基地址 (3MB偏移)

import ctypes
from ctypes import wintypes
import sys

import socket
import struct
import torch
import numpy as np
# import LN_RBN_BP_tch as rln
import time
from statistics import mean

device = torch.device('cpu')

# Windows API常量
DIGCF_PRESENT = 0x00000002
DIGCF_DEVICEINTERFACE = 0x00000010
INVALID_HANDLE_VALUE = -1
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x00000080
FILE_BEGIN = 0
ERROR_INSUFFICIENT_BUFFER = 122
MAX_PATH = 260

# XDMA GUID: 74c7e4a9-6d5d-4a70-bc0d-20691dff9e9d
GUID_DEVINTERFACE_XDMA = ctypes.c_char * 16
guid_bytes = bytes([
    0xa9, 0xe4, 0xc7, 0x74,
    0x5d, 0x6d,
    0x70, 0x4a,
    0xbc, 0x0d, 0x20, 0x69, 0x1d, 0xff, 0x9e, 0x9d
])

# XDMA设备文件名
XDMA_FILE_H2C_0 = "\\h2c_0"
XDMA_FILE_C2H_0 = "\\c2h_0"
XDMA_FILE_USER = "\\user"

# 缓冲区大小
ONE_MB = 1024 * 1024

# Windows API结构体定义
class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8)
    ]

class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("InterfaceClassGuid", GUID),
        ("Flags", wintypes.DWORD),
        ("Reserved", ctypes.POINTER(ctypes.c_ulong))
    ]

class SP_DEVICE_INTERFACE_DETAIL_DATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("DevicePath", wintypes.WCHAR * (MAX_PATH + 1))
    ]

# 加载Windows API函数
setupapi = ctypes.windll.setupapi
kernel32 = ctypes.windll.kernel32

# SetupDiGetClassDevs
SetupDiGetClassDevs = setupapi.SetupDiGetClassDevsW
SetupDiGetClassDevs.argtypes = [
    ctypes.POINTER(GUID),
    wintypes.LPCWSTR,
    wintypes.HWND,
    wintypes.DWORD
]
SetupDiGetClassDevs.restype = wintypes.HANDLE

# SetupDiEnumDeviceInterfaces
SetupDiEnumDeviceInterfaces = setupapi.SetupDiEnumDeviceInterfaces
SetupDiEnumDeviceInterfaces.argtypes = [
    wintypes.HANDLE,
    ctypes.c_void_p,
    ctypes.POINTER(GUID),
    wintypes.DWORD,
    ctypes.POINTER(SP_DEVICE_INTERFACE_DATA)
]
SetupDiEnumDeviceInterfaces.restype = wintypes.BOOL

# SetupDiGetDeviceInterfaceDetail
SetupDiGetDeviceInterfaceDetail = setupapi.SetupDiGetDeviceInterfaceDetailW
SetupDiGetDeviceInterfaceDetail.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(SP_DEVICE_INTERFACE_DATA),
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p
]
SetupDiGetDeviceInterfaceDetail.restype = wintypes.BOOL

# SetupDiDestroyDeviceInfoList
SetupDiDestroyDeviceInfoList = setupapi.SetupDiDestroyDeviceInfoList
SetupDiDestroyDeviceInfoList.argtypes = [wintypes.HANDLE]
SetupDiDestroyDeviceInfoList.restype = wintypes.BOOL

# CreateFile
CreateFile = kernel32.CreateFileW
CreateFile.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE
]
CreateFile.restype = wintypes.HANDLE

# CloseHandle
CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

# ReadFile
ReadFile = kernel32.ReadFile
ReadFile.argtypes = [
    wintypes.HANDLE,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p
]
ReadFile.restype = wintypes.BOOL

# WriteFile
WriteFile = kernel32.WriteFile
WriteFile.argtypes = [
    wintypes.HANDLE,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p
]
WriteFile.restype = wintypes.BOOL

# SetFilePointer
SetFilePointer = kernel32.SetFilePointer
SetFilePointer.argtypes = [
    wintypes.HANDLE,
    wintypes.LONG,
    ctypes.POINTER(wintypes.LONG),
    wintypes.DWORD
]
SetFilePointer.restype = wintypes.DWORD

# GetLastError
GetLastError = kernel32.GetLastError
GetLastError.argtypes = []
GetLastError.restype = wintypes.DWORD


class XDMADevice:
    """XDMA设备类，管理PCIe设备连接和数据传输"""
    
    def __init__(self):
        self.base_path = None
        self.h2c0_handle = None  # Host to Card 句柄
        self.c2h0_handle = None  # Card to Host 句柄
        self.user_handle = None  # User 句柄 (AXI Lite)
        self.buffer_c2h = None   # Card to Host 缓冲区
        self.buffer_h2c = None   # Host to Card 缓冲区
        self.buf_c2h_size = ONE_MB
        self.buf_h2c_size = ONE_MB
        
    def find_devices(self):
        """查找XDMA设备"""
        # 创建GUID
        guid = GUID()
        guid.Data1 = 0x74c7e4a9
        guid.Data2 = 0x6d5d
        guid.Data3 = 0x4a70
        guid.Data4 = (ctypes.c_ubyte * 8)(0xbc, 0x0d, 0x20, 0x69, 0x1d, 0xff, 0x9e, 0x9d)
        
        print(f"正在查找XDMA设备 (GUID: 74c7e4a9-6d5d-4a70-bc0d-20691dff9e9d)...")
        
        # 获取设备信息
        device_info = SetupDiGetClassDevs(
            ctypes.byref(guid),
            None,
            None,
            DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
        )
        
        if device_info == INVALID_HANDLE_VALUE or device_info == 0:
            error_code = GetLastError()
            print(f"SetupDiGetClassDevs失败，错误码: {error_code}")
            print("可能的原因:")
            print("  1. XDMA驱动未安装")
            print("  2. FPGA设备未连接")
            print("  3. 需要管理员权限")
            return 0
        
        device_interface = SP_DEVICE_INTERFACE_DATA()
        device_interface.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
        
        device_count = 0
        index = 0
        
        #print(f"开始枚举设备...")
        
        # 枚举设备
        while SetupDiEnumDeviceInterfaces(
            device_info,
            None,
            ctypes.byref(guid),
            index,
            ctypes.byref(device_interface)
        ):
            print(f"找到设备接口 #{index}")
            
            # 获取所需缓冲区大小
            detail_length = wintypes.DWORD(0)
            SetupDiGetDeviceInterfaceDetail(
                device_info,
                ctypes.byref(device_interface),
                None,
                0,
                ctypes.byref(detail_length),
                None
            )
            
            last_error = GetLastError()
            if last_error != ERROR_INSUFFICIENT_BUFFER:
                print(f"  获取设备详情失败，错误码: {last_error}")
                break
            
            # 分配并获取设备详细信息
            detail_size = detail_length.value
            detail_buffer = ctypes.create_string_buffer(detail_size)
            dev_detail = ctypes.cast(detail_buffer, ctypes.POINTER(SP_DEVICE_INTERFACE_DETAIL_DATA))
            
            # ⚠️ 重要：Windows API要求cbSize必须是特定值，不是结构体大小！
            # 这是Windows API的已知特性/bug
            # 64位系统: cbSize = 8
            # 32位系统: cbSize = 6
            import struct
            if struct.calcsize("P") == 8:  # 64位系统
                dev_detail.contents.cbSize = 8
            else:  # 32位系统
                dev_detail.contents.cbSize = 6
            
            if SetupDiGetDeviceInterfaceDetail(
                device_info,
                ctypes.byref(device_interface),
                dev_detail,
                detail_size,
                None,
                None
            ):
                if index == 0:  # 保存第一个找到的设备路径
                    self.base_path = dev_detail.contents.DevicePath
                    print(f"Found XDMA device: {self.base_path}")
                device_count += 1
            
            index += 1
        
        SetupDiDestroyDeviceInfoList(device_info)
        
        if device_count == 0:
            print("\n未找到XDMA设备！")
            print("\n请检查:")
            print("  1. FPGA板卡是否正确连接到PCIe插槽")
            print("  2. Xilinx XDMA驱动是否已安装")
            print("  3. 设备管理器中是否能看到XDMA设备")
            print("  4. 是否以管理员身份运行此程序")
        
        return device_count
    
    def open_device(self, device_name, access_mode):
        """打开指定的设备"""
        if not self.base_path:
            print("Error: Device path not found")
            return None
        
        device_path = self.base_path + device_name
        
        handle = CreateFile(
            device_path,
            access_mode,
            0,  # 不共享
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None
        )
        
        if handle == INVALID_HANDLE_VALUE:
            error_code = GetLastError()
            print(f"Error opening device: {device_path}, error code: {error_code}")
            return None
        
        return handle
    
    def init(self):
        """初始化PCIe设备"""
        # 查找设备
        num_devices = self.find_devices()
        #print(f"Found {num_devices} XDMA device(s).")
        
        if num_devices < 1:
            print("Error: No XDMA devices found")
            return -1
        
        # 分配缓冲区
        self.buffer_c2h = ctypes.create_string_buffer(self.buf_c2h_size)
        self.buffer_h2c = ctypes.create_string_buffer(self.buf_h2c_size)
        
        if not self.buffer_c2h:
            print("Error allocating buffer")
            return -1
        
        #print("PCIe initialization successful")
        return 1
    
    def open(self):
        """打开PCIe设备"""
        # 打开 Host-to-Card 0 设备
        self.h2c0_handle = self.open_device(
            XDMA_FILE_H2C_0,
            GENERIC_READ | GENERIC_WRITE
        )
        
        if not self.h2c0_handle or self.h2c0_handle == INVALID_HANDLE_VALUE:
            print("Error opening H2C device")
            return -1
        
        # 打开 Card-to-Host 0 设备
        self.c2h0_handle = self.open_device(
            XDMA_FILE_C2H_0,
            GENERIC_READ | GENERIC_WRITE
        )
        
        if not self.c2h0_handle or self.c2h0_handle == INVALID_HANDLE_VALUE:
            print("Error opening C2H device")
            if self.h2c0_handle:
                CloseHandle(self.h2c0_handle)
            return -1
        
        # 打开 User 设备 (AXI Lite)
        self.user_handle = self.open_device(
            XDMA_FILE_USER,
            GENERIC_READ | GENERIC_WRITE
        )
        
        if not self.user_handle or self.user_handle == INVALID_HANDLE_VALUE:
            print("Error opening User device (AXI Lite)")
            if self.h2c0_handle:
                CloseHandle(self.h2c0_handle)
            if self.c2h0_handle:
                CloseHandle(self.c2h0_handle)
            return -1
        
        #print("PCIe devices opened successfully (H2C, C2H, User)")
        return 1
    
    def close(self):
        """关闭PCIe设备"""
        if self.c2h0_handle:
            CloseHandle(self.c2h0_handle)
            self.c2h0_handle = None
        
        if self.h2c0_handle:
            CloseHandle(self.h2c0_handle)
            self.h2c0_handle = None
        
        if self.user_handle:
            CloseHandle(self.user_handle)
            self.user_handle = None
        
        print("PCIe devices closed")
    
    def write_device(self, handle, offset, data):
        """向设备写入数据"""
        if not handle or handle == INVALID_HANDLE_VALUE:
            print("Error: Invalid device handle")
            return -1
        
        # 设置文件指针
        result = SetFilePointer(handle, offset, None, FILE_BEGIN)
        if result == 0xFFFFFFFF:  # INVALID_SET_FILE_POINTER
            error_code = GetLastError()
            print(f"Error setting file pointer, error code: {error_code}")
            return -1
        
        # 写入数据
        bytes_written = wintypes.DWORD(0)
        
        # 确保data是bytes类型
        if isinstance(data, str):
            data = data.encode()
        elif isinstance(data, list):
            data = bytes(data)
        
        size = len(data)
        
        success = WriteFile(
            handle,
            data,
            size,
            ctypes.byref(bytes_written),
            None
        )
        
        if not success:
            error_code = GetLastError()
            print(f"WriteFile failed with error code: {error_code}")
            return -1
        
        return bytes_written.value
    
    def read_device(self, handle, offset, size):
        """从设备读取数据"""
        if not handle or handle == INVALID_HANDLE_VALUE:
            print("Error: Invalid device handle")
            return None
        
        # 设置文件指针
        result = SetFilePointer(handle, offset, None, FILE_BEGIN)
        if result == 0xFFFFFFFF:  # INVALID_SET_FILE_POINTER
            error_code = GetLastError()
            print(f"Error setting file pointer, error code: {error_code}")
            return None
        
        # 读取数据
        buffer = ctypes.create_string_buffer(size)
        bytes_read = wintypes.DWORD(0)
        
        success = ReadFile(
            handle,
            buffer,
            size,
            ctypes.byref(bytes_read),
            None
        )
        
        if not success:
            error_code = GetLastError()
            print(f"ReadFile failed with error code: {error_code}")
            return None
        
        return bytes(buffer[:bytes_read.value])
    
    def h2c_transfer(self, offset, data):
        """
        Host到Card的数据传输（写入FPGA）
        
        参数:
            offset: 目标地址偏移
            data: 要写入的数据（bytes或list）
        
        返回:
            成功写入的字节数，失败返回-1
        """
        return self.write_device(self.h2c0_handle, offset, data)
    
    def c2h_transfer(self, offset, size):
        """
        Card到Host的数据传输（从FPGA读取）
        
        参数:
            offset: 源地址偏移
            size: 要读取的字节数
        
        返回:
            读取到的数据（bytes），失败返回None
        """
        # 如果size小于5，先读取10字节再截取（与C代码逻辑一致）
        if size < 5:
            temp_data = self.read_device(self.c2h0_handle, offset, 10)
            if temp_data:
                return temp_data[:size]
            return None
        else:
            return self.read_device(self.c2h0_handle, offset, size)
    
    def write_user(self, offset, value):
        """
        写入AXI Lite用户空间寄存器（GPIO等外设）
        
        参数:
            offset: 寄存器地址偏移
            value: 要写入的32位值（uint32）
        
        返回:
            成功写入的字节数，失败返回-1
        """
        if not self.user_handle or self.user_handle == INVALID_HANDLE_VALUE:
            print("Error: User device not opened")
            return -1
        
        # 将value转换为4字节（uint32）
        import struct
        data = struct.pack('<I', value)  # 小端序，无符号32位整数
        
        return self.write_device(self.user_handle, offset, data)
    
    def read_user(self, offset):
        """
        读取AXI Lite用户空间寄存器（GPIO等外设）
        
        参数:
            offset: 寄存器地址偏移
        
        返回:
            读取到的32位值（uint32），失败返回None
        """
        if not self.user_handle or self.user_handle == INVALID_HANDLE_VALUE:
            print("Error: User device not opened")
            return None
        
        # 读取4字节
        data = self.read_device(self.user_handle, offset, 4)
        
        if data and len(data) == 4:
            # 将4字节转换为uint32（小端序）
            import struct
            value = struct.unpack('<I', data)[0]
            return value
        
        return None


# 全局设备实例
xdma_device = XDMADevice()


def pcie_init():
    """初始化PCIe设备"""
    return xdma_device.init()


def pcie_open():
    """打开PCIe设备"""
    return xdma_device.open()


def pcie_close():
    """关闭PCIe设备"""
    xdma_device.close()


def h2c_transfer(offset, data):
    """
    Host到Card的数据传输（写入FPGA）
    
    参数:
        offset: 目标地址偏移
        data: 要写入的数据（bytes、str或list）
    
    返回:
        成功写入的字节数，失败返回-1
    """
    return xdma_device.h2c_transfer(offset, data)


def c2h_transfer(offset, size):
    """
    Card到Host的数据传输（从FPGA读取）
    
    参数:
        offset: 源地址偏移
        size: 要读取的字节数
    
    返回:
        读取到的数据（bytes），失败返回None
    """
    return xdma_device.c2h_transfer(offset, size)


def write_user(offset, value):
    """
    写入AXI Lite用户空间寄存器（GPIO等外设）
    
    参数:
        offset: 寄存器地址偏移
        value: 要写入的32位值（uint32，0-4294967295）
    
    返回:
        成功写入的字节数，失败返回-1
    
    示例:
        # 向偏移0写入LED控制值
        write_user(0, 0x0F)  # 点亮LED 0-3
        
        # 向偏移4写入其他寄存器值
        write_user(4, 0x12345678)
    """
    return xdma_device.write_user(offset, value)


def read_user(offset):
    """
    读取AXI Lite用户空间寄存器（GPIO等外设）
    
    参数:
        offset: 寄存器地址偏移
    
    返回:
        读取到的32位值（uint32），失败返回None
    
    示例:
        # 从偏移8读取按键值
        key_value = read_user(8)
        if key_value is not None:
            print(f"Key value: 0x{key_value:08X}")
            
            # 检查某个位
            if key_value & 0x01:
                print("Bit 0 is set")
    """
    return xdma_device.read_user(offset)


def check_xdma_driver():
    """检查XDMA驱动是否安装"""
    import subprocess
    try:
        # 使用PowerShell查询PCI设备
        #print("\n正在检查系统中的PCIe设备...")
        cmd = 'Get-PnpDevice -Class "System devices" | Where-Object {$_.FriendlyName -like "*XDMA*" -or $_.FriendlyName -like "*Xilinx*"} | Format-Table -AutoSize'
        result = subprocess.run(
            ["powershell", "-Command", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=10
        )
        
        if result.stdout.strip():
            print("找到以下Xilinx/XDMA相关设备:")
            print(result.stdout)
        # 不显示"未找到设备"的消息，因为程序会通过其他方式检测设备
        
    except Exception as e:
        print(f"检查设备时出错: {e}")

def write_fp16_array_to_ddr(data, ddr_address, batch_size=128, verify=True):
    """
    将FP16数组分批写入DDR
    data: FP32数组
    ddr_address: DDR起始地址
    batch_size: 每批传输的元素个数
    verify: 是否读回验证
    返回: (success_count, total_batches)
    """
    import numpy as np
    
    # FP32 -> FP16 -> bytes
    if isinstance(data, (list, tuple)):
        data_fp32 = np.array(data, dtype=np.float32)
    elif isinstance(data, np.ndarray):
        data_fp32 = data.astype(np.float32)
    else:
        return (0, 0)
    
    data_fp16 = data_fp32.astype(np.float16)
    fp16_bytes = data_fp16.tobytes()
    
    # 分批传输
    total_batches = (len(fp16_bytes) + batch_size * 2 - 1) // (batch_size * 2)
    success_count = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size * 2
        end_idx = min(start_idx + batch_size * 2, len(fp16_bytes))
        batch_data = fp16_bytes[start_idx:end_idx]
        
        # 写入DDR
        bytes_written = h2c_transfer(ddr_address + start_idx, batch_data)
        if bytes_written <= 0:
            continue
        
        # 可选：读回验证
        if verify:
            read_back = c2h_transfer(ddr_address + start_idx, len(batch_data))
            if read_back and read_back == batch_data:
                success_count += 1
        else:
            success_count += 1
    
    return (success_count, total_batches)


# 示例使用
if __name__ == "__main__":
    import sys
       
    # 首先检查驱动
    check_xdma_driver()
    
    # 初始化设备
    if pcie_init() < 0:
        print("初始化失败")
        sys.exit(1)
    
    # 打开设备
    if pcie_open() < 0:
        print("打开设备失败")
        sys.exit(1)
    
    try:
        import numpy as np
        
        print(f"\n{'='*60}")
        print(f"模型参数传输: Seq={X_DIM}, Hidden={Y_DIM}")
        print(f"{'='*60}")
        
        # ====== 1. 准备X数据 (sequence_length x hidden_states) ======
        print(f"\n[1/4] X数据 ({X_DIM}x{Y_DIM} = {X_DIM*Y_DIM}个元素)")
        x_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        x_data = np.array([x_values[i % 8] for i in range(X_DIM * Y_DIM)], dtype=np.float32)
        print(f"  数据: {x_values[:8]} (循环)")
        print(f"  地址: 0x{ADDR_X_DATA:08X}")
        
        success, total = write_fp16_array_to_ddr(x_data, ADDR_X_DATA, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        
        # ====== 2. 准备权重 (hidden_states个) ======
        print(f"\n[2/4] 权重 ({Y_DIM}个元素)")
        # 固定权重值：0.5
        weights = np.full(Y_DIM, 0.5, dtype=np.float32)
        print(f"  数据: 全部为 0.5")
        print(f"  地址: 0x{ADDR_WEIGHTS:08X}")
        
        success, total = write_fp16_array_to_ddr(weights, ADDR_WEIGHTS, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        
        # ====== 3. 准备偏置 (hidden_states个) ======
        print(f"\n[3/4] 偏置 ({Y_DIM}个元素)")
        # 固定偏置值：0.1
        bias = np.full(Y_DIM, 0.1, dtype=np.float32)
        print(f"  数据: 全部为 0.1")
        print(f"  地址: 0x{ADDR_BIAS:08X}")
        
        success, total = write_fp16_array_to_ddr(bias, ADDR_BIAS, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        
        # ====== 4. 准备梯度 (sequence_length x hidden_states) ======
        print(f"\n[4/4] 梯度 ({X_DIM}x{Y_DIM} = {X_DIM*Y_DIM}个元素)")
        # 固定梯度值：1.0
        grad = np.full(X_DIM * Y_DIM, 1.0, dtype=np.float32)
        print(f"  数据: 全部为 1.0")
        print(f"  地址: 0x{ADDR_GRAD:08X}")
        
        success, total = write_fp16_array_to_ddr(grad, ADDR_GRAD, BATCH_SIZE)
        print(f"  传输: {success}/{total} 批次", "✓" if success == total else "❌")
        
        # ====== 总结 ======
        print(f"\n{'='*60}")
        print(f"传输完成")
        print(f"  X数据:  0x{ADDR_X_DATA:08X} ({X_DIM*Y_DIM}个FP16, {X_DIM*Y_DIM*2}字节)")
        print(f"  权重:   0x{ADDR_WEIGHTS:08X} ({Y_DIM}个FP16, {Y_DIM*2}字节)")
        print(f"  偏置:   0x{ADDR_BIAS:08X} ({Y_DIM}个FP16, {Y_DIM*2}字节)")
        print(f"  梯度:   0x{ADDR_GRAD:08X} ({X_DIM*Y_DIM}个FP16, {X_DIM*Y_DIM*2}字节)")
        print(f"{'='*60}")
        
    except ImportError:
        print("❌ 需要numpy库: pip install numpy")
        
    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        # 关闭设备
        pcie_close()
        print("\n程序结束")


