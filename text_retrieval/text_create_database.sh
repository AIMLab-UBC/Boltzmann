#!/bin/bash

# Activate Conda environment
#conda activate cuda12_4

# Run Python script
python3 ./text_retrieval/text_create_database.py \
    --task_id 0 \
    --path_to_features "./assets/LLM/Llama3-8b/text/" \
    --path_to_save "./assets/generated_files/database/text/" \
    --dataset TCGA \
    --LLM Llama3-8b \
    --num_arrays 1
