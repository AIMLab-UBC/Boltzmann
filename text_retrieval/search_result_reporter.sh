#!/bin/bash

conda activate cuda12_4

LLM=Llama3-8b
top_k=3
search_type=subtype
keyword=text

python ./text_retrieval/search_result_reporter.py \
    --path_to_csv ./assets/files/TCGA_Reports_with_metadata.csv \
    --path_to_search_jsons ./assets/generated_files/search_results/final_search/${keyword}/ \
    --path_to_save ./assets/generated_files/final_results/${keyword}/ \
    --top_k "$top_k" \
    --dataset TCGA \
    --LLM "$LLM" \
    --search_type "$search_type" \