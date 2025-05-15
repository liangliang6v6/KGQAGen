DATA_PATH=rmanluo
DATA_LIST="RoG-webqsp"
SPLIT="test"
# REASONING_PATH="results/GenPaths/${DATA}/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10-index_len2/predictions.jsonl"
# MODEL_NAME=gpt-3.5-turbo
# N_THREAD=10

MODEL_NAME=gpt-4o-mini
N_THREAD=10

for DATA in ${DATA_LIST}; do
  # REASONING_PATH="results/GenPaths/${DATA}/rmanluo/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10/predictions.jsonl"
  REASONING_PATH="results/GenPaths/${DATA}/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10-index_len2/predictions.jsonl"
  python workflow/predict_final_answer.py --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --model_name ${MODEL_NAME} --reasoning_path ${REASONING_PATH} --add_path True -n ${N_THREAD}
done
