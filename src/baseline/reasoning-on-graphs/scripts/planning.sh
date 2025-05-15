#!/bin/bash
SPLIT="test"
DATASET_LIST="QoG5" # RoG-cwq RoG-webqsp
MODEL_NAME=RoG
MODEL_PATH=/data/liang/workspace/KG/train/wiki

BEAM_LIST="5" # "1 2 3 4 5"
for DATASET in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        python src/qa_prediction/gen_rule_path.py \
        --model_name ${MODEL_NAME} \
        --model_path ${MODEL_PATH} \
        -d ${DATASET} \
        --split ${SPLIT} \
        --n_beam ${N_BEAM}
    done
done