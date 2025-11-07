# 面向模型训练的归一化层硬件设计

本仓库可复现“可重配置归一化（Reconfigurable Normalization, RN）训练加速系统”，围绕**块层归一化（Block Layer Normalization, BLN）**与**RMSNorm**两类算子展开软硬件协同优化。项目以 AMD Xilinx ZCU106 为目标平台，融合 PCIe/UDP 高速链路、PS 侧 FreeRTOS 协议栈与 PL 侧可重构向量处理阵列，实现归一化层的前向与反向传播全流程加速。

## 项目亮点

- **平方根消除的 BLN 近似算法**：通过按块统计范围值近似方差，规避根号运算及方差-均值串行依赖，在 FP16 精度下保持 1e-2 量级误差。
- **可重配置归一化电路**：在 BLN 数据通路上增加 RMSNorm 旁路，可按寄存器配置在两种模式间切换，实现一套硬件覆盖多种归一化需求。
- **端到端训练级验证**：提供基于 HuggingFace Transformers 的问答与文本分类测试脚本，可对比 CPU 参考实现，验证梯度正确性。
- **显著性能收益**：在 Intel i5-10500 对比下，BLN/RMSNorm 模式分别取得 201.28× 与 195.68× 端到端加速比。

## 系统架构速览

- **Host（PC 端）**：位于 `host/`，在 Windows 环境下运行，负责数据到 DDR 的 PCIe XDMA 传输、UDP 控制报文下发以及结果读回与精度/性能分析。
- **PS（ARM 端）**：位于 `vitis/`，基于 FreeRTOS + lwIP 的裸机应用，管理 UDP 指令解析、AXI CDMA 搬运、常数注入与结果写回。
- **PL（FPGA 逻辑）**：Vivado 工程位于 `vivado/`，包含 RN 可重配置算子、前向流水线与三个反向通路以及时钟周期统计单元。
- **Algo 验证**：`train/` 提供 PyTorch/Python 端的 BLN 近似算子精度验证。

## 目录结构

- `host/`：PC 侧 PCIe/UDP 上位机，与 PyTorch 参考实现比对。
- `train/`：基于 HuggingFace 的任务脚本，验证 RN 算子在问答与文本分类场景的数值与训练效果。
- `vitis/`：PS 核 C 工程源码，可直接导入 Vitis 构建 FreeRTOS 应用。
- `vivado/`：PL 端硬件工程（请在 Vivado 中打开），含 RN 加速器 IP 与系统连接。
- `Readme.md`：仓库总览（当前文件）。

各子目录内的 `README.md` 给出了更细致的依赖、编译和运行说明。

## 快速上手

1. **准备硬件**：ZCU106（或等效）开发板加载 RN bitstream，安装 Xilinx XDMA 驱动并配置板载网络 `192.168.1.10`。
2. **部署 PS 应用**：使用 Vitis 导入 `vitis/`，编译并烧写 FreeRTOS + RN 控制程序。
3. **运行 Host**：在 Windows PC 上安装 Python 3.8+、`torch`、`numpy` 等依赖，运行 `host/pcie_udp_test_bln.py` 或 `host/pcie_udp_test_rms.py` 即可完成数据写入、板端执行与结果验证。
4. **算法验证（可选）**：在 GPU/CPU 上运行 `train/` 下的脚本，对比原生 LayerNorm 与 BLN/RMSNorm 算子的训练表现。

## 性能与精度

- 精度：BLN/RMSNorm 两种模式在相对误差与绝对误差 `1e-2` 范围内与 PyTorch 保持一致。
- 性能：在 250 MHz 主频、FP16 精度配置下，分别获得 `201.28×`（BLN）与 `195.68×`（RMSNorm）的整体加速比。

## 更多信息

- 算法设计、测试流程、扩展建议等请查阅子目录 README。
- 若需在其他平台移植，请重点关注 `define.h` 中的尺寸、地址与常数配置。

欢迎提出改进意见，共同完善面向训练的归一化算子硬件方案。