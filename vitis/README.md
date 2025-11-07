# PS 端 Vitis 工程

`vitis/` 目录提供 AMD Xilinx Vitis 工程文件与源码，用于在 Zynq UltraScale+ PS 核上管理 BLN 归一化加速器的运行时逻辑。目录下已补齐 `sys_wrapper.xsa`，可直接用于创建应用项目。

## 功能摘要

- **网络通信**：`udp_perf_server.c`/`udp_func.h` 构建 lwIP UDP 服务，解析来自 PC 的激活命令并回复完成信号。
- **PCIe/CDMA 协调**：`PL_data_manage.c` 通过 AXI CDMA 在 PS DDR 与 PL 加速器之间搬运输入、权重、偏置与梯度数据，同时写入常数参数、读取结果与性能计数。
- **线程管理**：`main.c` 创建 `main_thread` 与 `main_app`，其中 `main_thread` 负责网络与 CDMA 初始化，`main_app` 轮询执行 `app_main()`。
- **配置常量**：`define.h` 集中定义序列长度、隐藏维度、FP16 参数、DDR 地址映射以及 CDMA/AXI-Lite 寄存器偏移。

## 构建步骤

1. 启动 Vitis，选择 **Create Application Project**，`Hardware platform` 直接指向本目录下的 `sys_wrapper.xsa`。
2. 在 `Domain` 设置中选择 **freertos**（含 lwIP）。
3. 选择示例 **lwIP UDP Perf Server** 作为模板，让 Vitis 自动生成基础 UDP 工程。
4. 将生成工程 `src/` 目录中的源文件，用本仓库 `vitis/src/` 下文件全量覆盖（可直接复制替换），以引入 BLN 控制逻辑。
5. 根据硬件设计调整 `platform_config.h` 中的 MAC、IP 等参数，并与 Host 端保持一致。
6. 如需更改归一化尺寸，请同步修改 `define.h` 与 Host 侧脚本中的常量。
7. 编译生成 `.elf` 后，通过 JTAG 或 SD 卡方式部署至板卡。

## 运行流程

1. FPGA 加载含 RN 加速器的 bitstream；
2. PS 端运行该应用，串口将输出网络初始化、CDMA 状态以及 FPGA 反馈信息；
3. Host 端脚本通过 UDP 激活执行，PS 端收到命令后依次完成常数写入、权重/偏置/输入初始化、触发前向/反向流水线，并将结果写回 DDR；
4. 最终通过 UDP 返回完成信号，供 Host 端继续读回数据。

## 调试提示

- `PL_feedback` 的状态机值可通过串口日志观察；
- `PL_data_manage.c` 中提供 `xil_printf` 打印，可按需开启以定位 CDMA 传输问题；
- 若使用中断模式，需要在 `define.h` 中启用 `ENABLE_CDMA` 并确认 GIC 号与硬件设计一致。

更多系统级说明请参考仓库根目录 README 及 Host/Vivado 文档。

