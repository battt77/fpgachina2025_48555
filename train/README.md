# 算法验证与任务测试

`train/` 目录提供基于 HuggingFace Transformers 的示例脚本，用于验证 BLN/RMSNorm 算子在主流 NLP 任务（问答、文本分类）中的数值稳定性与训练表现。所有脚本均保留官方实现的参数体系，并额外集成 `LN_RBN_BP_tch.py` 中的自定义归一化模块。

## 目录结构

- `question_answering/`：SQuAD 等数据集上的问答任务脚本。
- `text_classification/`：GLUE/IMDb 等数据集上的分类脚本。
- 每个子目录包含：
  - 适配 RN 算子的 `run_*_rln.py`/`run_*_no_trainer.py` 等脚本；
  - 与官方实现保持一致的原始脚本，便于对照；
  - `requirements.txt` 与示例启动脚本（`.sh`）。

## 环境准备

- Python 3.8+，建议使用 Conda 或 virtualenv。
- 安装 Requirements：

```bash
pip install -r question_answering/requirements.txt
pip install -r text_classification/requirements.txt
pip install transformers==4.45.0.dev0
```

若需使用 GPU，请确保安装对应版本的 PyTorch 与 CUDA。

## 使用方法概述

1. **选择任务**：进入目标子目录（如 `question_answering`）。
2. **准备数据**：默认使用 🤗 Datasets 自动下载的公开数据集；若需自定义数据，可参考脚本中的数据预处理注释。
3. **运行脚本**：

```bash
python run_qa_rln.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train --do_eval \
  --output_dir outputs/rln_qa
```

脚本会自动在模型中插入 BLN/RMSNorm 算子，并对比 PyTorch 原生 LayerNorm 的结果。

4. **对比实验**：可分别运行原始 `run_qa.py` 与 `run_qa_rln.py`，比较收敛速度、精度及日志中的归一化层调用情况。

## 与硬件联动

虽然这些脚本主要用于离线验证，但输出的模型/梯度数据可作为 Host 侧测试脚本的输入来源，用于构造更贴近真实训练场景的验证数据。常见做法包括：

- 在训练前几步保存中间张量，通过 `pcie_rw.write_fp16_array_to_ddr` 推送到 FPGA；
- 使用自定义 Autograd Function 生成梯度参考，与 PL 侧输出比对。

## 注意事项

- 运行大型模型可能需要较大显存，请根据硬件资源调整 `per_device_train_batch_size`、`max_seq_length` 等参数。
- 默认日志使用标准输出，可结合 `--logging_steps` 与 `--save_strategy` 控制训练记录与模型保存频率。
- 若使用 `accelerate` 启动多卡训练，请先运行 `accelerate config` 并根据提示配置。

更多关于具体任务的说明，请参见对应子目录的 README。

