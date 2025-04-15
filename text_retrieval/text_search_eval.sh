#!/bin/bash


conda activate cuda12_4


model="Llama3-8b"
mode="subtype"  #type for disease search in organ-independent setting and subtype for organ-specific setting
keyword="text"

python3 ./text_retrieval/text_search_eval.py \
    --path_to_database ./assets/generated_files/database/${keyword}/${model}/ \
    --path_to_save ./assets/generated_files/search_results/final_search/${keyword}/ \
    --path_to_csv ./assets/files/TCGA_Reports_with_metadata.csv \
    --path_to_cancer_map ./assets/files/cancer_oragn_map.json \
    --dataset TCGA \
    --LLM Llama3-8b \
    --top_k 10 \
    --search_type subtype \
    --task_id 0 \
    --num_arrays 1 \