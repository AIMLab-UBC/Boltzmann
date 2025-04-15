#!/bin/bash

conda activate cuda12_4

LLM=Llama3-8b
dataset=LGG

if [ "$LLM" == "Coral" ] ; then
  c_size=8192
elif [ "$LLM" == "Gemma-7b" ]; then
  c_size=3072
elif [ "$LLM" == "Jamba" ]; then
  c_size=1024
elif [ "$LLM" == "Llama3-8b" ] ||  [ "$LLM" == "Bio-Llama3-8b" ]; then
  c_size=4096
fi

#UNI PLIP CTransPath Phikon Lunit-Dino swin vit
python ./boltzmann_semantic_score/vision_language_score_evaluator.py \
    --cancer_types "$dataset" \
    --search_target "$dataset" \
    --vis_encoders UNI \
    --LLM  "$LLM" \
    --text_feat_size "$c_size" \
    --top_k 1 3 5 10 15 \
    --visual_feat_dir ./assets/LVM/ \
    --text_feat_dir ./assets/generated_files/database/text/ \
    --text_manifest_path ./assets/files/TCGA_Reports_with_metadata.csv \
    --path_to_save ./assets/generated_files/BSS/ \