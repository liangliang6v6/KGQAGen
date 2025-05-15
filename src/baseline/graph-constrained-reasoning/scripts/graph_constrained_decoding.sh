DATA_PATH=lianglz
DATA_LIST="KGQAGen-10k"
SPLIT="test"
INDEX_LEN=2
ATTN_IMP=flash_attention_2

MODEL_PATH=/data/liang/workspace/KG/train/GCR-wiki-1
MODEL_NAME=$(basename "$MODEL_PATH")

K="3" # 3 5 10 20
for DATA in ${DATA_LIST}; do
  for k in $K; do
    python workflow/predict_paths_and_answers.py --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --index_path_length ${INDEX_LEN} --model_name ${MODEL_NAME} --model_path ${MODEL_PATH} --k ${k} --prompt_mode zero-shot --generation_mode group-beam --attn_implementation ${ATTN_IMP}
  done
done
