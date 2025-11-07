# PL 端 Vivado 工程

`vivado/` 目录用于存放 BLN 归一化加速电路的 Vivado 工程文件。请使用本目录提供的 TCL 脚本快速搭建工程，并确保 IP 仓库与项目文件位于同级目录。

## 建议的工程内容

- **顶层 block design**：包含 BLN 加速器自定义 IP、AXI XDMA、AXI CDMA、AXI Lite 控制接口、DDR/HP 端口、时钟与复位模块。
- **归一化加速 IP**：实现 BLN/RMSNorm 双模式流水线，支持前向输出、输入梯度、权重/偏置梯度及时钟计数输出。
- **时钟规划**：推荐 250 MHz 主时钟，与 Host/Vitis 工程中的性能统计保持一致。
- **地址映射**：确保与 `define.h`、`pcie_rw.py` 中的 DDR/AXI 地址表一致。

## 使用指引

1. 将提供的 `ip_repo/` 文件夹置于 `vivado/` 目录下（保持与工程同级）。
2. 打开 Vivado，**先**通过 `Tools -> Settings -> IP -> Repository` 添加本目录的 `ip_repo/`，或在 TCL 控制台执行：
   ```tcl
   set_property ip_repo_paths [list "${origin_dir}/ip_repo"] [current_project]
   update_ip_catalog
   ```
3. 在 Vivado TCL 控制台切换到工程目录，并指定 ZCU106 板卡：
   ```tcl
   cd $origin_dir
   ```
4. 依次执行以下脚本生成工程与 Block Design（脚本内部默认使用 `xczu7ev-ffvc1156-2-i` 及 ZCU106 板包）：
   ```tcl
   source hfn_v3_0_project.tcl
   source block_design.tcl
   ```
5. 脚本完成后即可在 GUI 中打开生成的工程，检查连接并根据需要修改参数。
6. 生成 bitstream 后，将 `.bit` 输出至 `vitis/` 平台项目或直接加载到板卡。
7. 若需导出硬件平台给 Vitis，请使用 `File -> Export -> Export Hardware`（包含 bitstream）。

## 版本控制建议

- 建议将 `*.xpr`、`sources/`、`bd/` 等关键文件纳入版本控制；
- 对于体积较大的 `runs/`、`sim/` 输出，可使用 `.gitignore` 排除；
- 若项目需分享给他人复现，请提供导出的 `xsa` 或 `dcp` 文件，并在此 README 中说明生成步骤。

如需更新仓库结构或添加硬件模块，请同步修改 Host/PS 侧的常量配置。

