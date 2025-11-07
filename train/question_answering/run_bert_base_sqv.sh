#!/bin/bash

task="squadv1_base"
dataset_name="squad"
model="bert"
model_name_or_path="bert-base-uncased"
max_seq_length=384
per_device_train_batch_size=48
learning_rate=3e-5
num_train_epochs=5
doc_stride=128
save_total_limit=2

# 标准 LayerNorm 对照实验
output_dir="../output/$model/$task/ln"
torchrun --nproc_per_node=4 run_qa.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name\
  --do_train \
  --do_eval \
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --doc_stride $doc_stride\
  --save_total_limit $save_total_limit\

'''
ln_batch=4
output_dir="../output/$model/$task/rln_$ln_batch"
torchrun --nproc_per_node=4 run_qa_rln.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name\
  --ln_batch  $ln_batch\
  --do_train \
  --do_eval \
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --doc_stride $doc_stride\
  --save_total_limit $save_total_limit\

ln_batch=8
output_dir="../output/$model/$task/rln_$ln_batch"
torchrun --nproc_per_node=4 run_qa_rln.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name\
  --ln_batch  $ln_batch\
  --do_train \
  --do_eval \
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --doc_stride $doc_stride\
  --save_total_limit $save_total_limit\

ln_batch=16
output_dir="../output/$model/$task/rln_$ln_batch"
torchrun --nproc_per_node=4 run_qa_rln.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name\
  --ln_batch  $ln_batch\
  --do_train \
  --do_eval \
  --fp16 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --output_dir $output_dir\
  --doc_stride $doc_stride\
  --save_total_limit $save_total_limit
'''


