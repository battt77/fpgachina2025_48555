# Host 端上位机

`host/` 目录提供 Windows 平台的控制程序，用于在 PC 端驱动 BLN 归一化加速电路，完成数据搬运、触发执行与结果校验的闭环流程。

## 功能概览

- **PCIe XDMA 传输**：`pcie_rw.py` 基于 Windows API 查找 Xilinx XDMA 设备，实现 Host↔DDR 的 FP16 批量写入/读回以及 AXI-Lite 寄存器访问。
- **UDP 控制链路**：`udp_client.py` 与板端 PS 应用的 lwIP UDP 服务通信，完成启动信号、状态查询与调试数据收发。
- **PyTorch 参考实现**：`LN_RBN_BP_tch.py` 与 `RMS_BP_tch.py` 内含 BLN/RMSNorm 的自定义 Autograd Function，可在主机侧生成前向/反向参考结果。
- **端到端验证脚本**：`pcie_udp_test_bln.py`、`pcie_udp_test_rms.py` 负责随机生成数据→PCIe 写入→UDP 触发→读回结果→与 PyTorch 比对→性能统计；`pcie_udp_test.py` 为通用调试入口。

## 环境准备

- Windows 10/11，已安装 Xilinx XDMA 驱动（建议管理员权限运行）。
- Python 3.8 及以上，建议创建虚拟环境。
- 主要依赖：`torch`、`numpy`、`statistics`、`ctypes`（标准库）等，可参考以下命令：

```bash
pip install torch numpy
```

## 典型流程

1. **确认硬件联通**：`pcie_rw.check_xdma_driver()` 可检测 XDMA 设备；运行 `udp_simple.py` 验证 UDP 报文收发。
2. **数据写入与参考计算**：执行 `pcie_udp_test_bln.py`（或 `pcie_udp_test_rms.py`），脚本会将输入、权重、偏置、输出梯度写入 DDR，同时调用 PyTorch 模型生成对照结果。
3. **触发板端执行**：脚本通过 UDP 发送激活命令，等待 PL 完成前向和反向计算。
4. **结果读回与比对**：通过 `c2h_transfer` 读取前向输出/各类梯度，并使用 `torch.allclose` 进行精度校验，最后打印 CPU 与 FPGA 的耗时与加速比。

## 常用脚本说明

- `pcie_udp_test_bln.py`：面向 BLN 模式的全流程验证，默认运行 10 次迭代并输出平均性能。
- `pcie_udp_test_rms.py`：将控制命令切换至 RMSNorm 模式，其余流程一致。
- `pcie_udp_test.py`：保留示例主流程，可根据需要裁剪或嵌入新的触发逻辑。
- `udp_simple.py`：最小化 UDP 测试脚本。

运行脚本前，请确保：

- `BOARD_IP`/`LOCAL_IP`、端口与板端配置一致。
- `pcie_rw.py` 中的 DDR 地址映射与 `define.h` 保持同步。
- 目标 bitstream 已部署到 FPGA，且 PS 应用正在运行。

## 调试建议

- 使用 `write_fp16_array_to_ddr` 的 `verify=False` 可跳过读回验证以提高写入速度。
- 若遇到 `INVALID_HANDLE_VALUE`，请确认 XDMA 设备驱动与管理员权限。
- UDP 超时通常意味着 PS 端未发送完成信号，可查看串口日志确认状态或检查 PL 寄存器回写。

更多软硬件联调细节可参考仓库根目录及 PS/Vivado 子目录的 README。

