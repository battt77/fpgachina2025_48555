#!/bin/bash

task="qnli"
model_name_or_path="bert-base-cased"
max_seq_length=128
per_device_train_batch_size=96
learning_rate=2e-5
num_train_epochs=6
save_total_limit=2

# 标准 LayerNorm 对照实验
output_dir="../output/bert/$task/ln"
torchrun --nproc_per_node=4 run_glue.py \
  --model_name_or_path $model_name_or_path \
  --task_name $task\
  --do_train \
  --do_eval \
  --do_predict\
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --save_total_limit $save_total_limit\

ln_batch=4
output_dir="../output/bert/$task/rln_$ln_batch"
torchrun --nproc_per_node=4 run_glue_rln.py \
  --model_name_or_path $model_name_or_path \
  --ln_batch  $ln_batch\
  --task_name $task\
  --do_train \
  --do_eval \
  --do_predict\
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --save_total_limit $save_total_limit\

ln_batch=8
output_dir="../output/bert/$task/rln_$ln_batch"
torchrun --nproc_per_node=4 run_glue_rln.py \
  --model_name_or_path $model_name_or_path \
  --ln_batch  $ln_batch\
  --task_name $task\
  --do_train \
  --do_eval \
  --do_predict\
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --save_total_limit $save_total_limit\

ln_batch=16
output_dir="../output/bert/$task/rln_$ln_batch"
torchrun --nproc_per_node=4 run_glue_rln.py \
  --model_name_or_path $model_name_or_path \
  --ln_batch  $ln_batch\
  --task_name $task\
  --do_train \
  --do_eval \
  --do_predict\
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --save_total_limit $save_total_limit\



