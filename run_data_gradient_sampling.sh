#!/usr/bin/env bash
set -e  # 有错误直接退出，防止“假跑完”

# ====== 配置区 ======
DATASETS=("tqa" "nq" "hqa" ) #"tqa" "nq"
MODELS=("llama3.1-8b-instruct"  ) #"qwen2.5-7b-instruct" "gemma2-9b-it" 
# DELTAS=(1.5 2.0 2.5 3.0 3.5 4.0 5.0)
DELTAS=(0.0)

GRAD_SCRIPT="collect_hidden_states/data_gradient_sampling_1_entloss.py"

PYTHON=python  # 或 python3
# ===================

echo "Start running experiments..." 
for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    # 如果是 hqa + llama3.1-8b-instruct 就跳过
    # if [[ "$DATASET" == "hqa" && "$MODEL" == "qwen2.5-7b-instruct" ]]; then
    #   echo "Skip: $DATASET + $MODEL"
    #   continue
    # fi
    for DELTA in "${DELTAS[@]}"; do
      echo "======================================"
      echo "Dataset: $DATASET | Model: $MODEL"
      echo "======================================"

      echo "Running data_gradients_sampling.py"
      $PYTHON $GRAD_SCRIPT \
        --model_path "/home/wangyujing/llm-aware/models/$MODEL" \
        --save_dir_name "/home/wangyujing/llm-aware/llm-aware-73/cashed_data_sampling_$DATASET"\
        --seeds_to_encode 42 \
        --data_name "$DATASET" \
        --k_shot 0\
        --delta $DELTA\
        --acc_threshold 0.8
        

      echo "Done: $DATASET + $MODEL"
      echo
    done
  done
done

echo "All experiments finished"
