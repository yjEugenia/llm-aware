#!/usr/bin/env bash
set -e  # 有错误直接退出，防止“假跑完”

# ====== 配置区 ======
DATASETS=("nq")
MODELS=("qwen2.5-7b-instruct" "gemma2-9b-it") #"llama3.1-8b-instruct" "qwen2.5-7b-instruct" 

GROUP_SCRIPT="collect_hidden_states/data_group_sampling.py"

PYTHON=python  # 或 python3
# ===================

echo "Start running experiments..."

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "======================================"
    echo "Dataset: $DATASET | Model: $MODEL"
    echo "======================================"

    echo "Running data_group_sampling.py"
    $PYTHON $GROUP_SCRIPT \
      --model_path "/home/wangyujing/llm-aware/models/$MODEL" \
      --save_dir_name "/home/wangyujing/llm-aware/cashed_data_sampling_$DATASET"\
      --seeds_to_encode 42 \
      --data_name "$DATASET" \
      --k_shot 0\
      --num_sample 10

      

    echo "Done: $DATASET + $MODEL"
    echo
  done
done

echo "All experiments finished"
