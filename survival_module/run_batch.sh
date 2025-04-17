#!/bin/bash
#SBATCH --job-name submit
#SBATCH --cpus-per-task 1
#SBATCH --mem=10GB


#declare -a models=("Coral" "Jamba" "Gemma-7b" "Llama3-8b" "Bio-Llama3-8b")
#declare -a models=("Llama3-8b")
declare -a models=("CTransPath" "Lunit-Dino" "Phikon" "PLIP" "swin" "UNI" "vit")
declare -a dataset_names=("BRCA" "GBM" "KIRC" "KIRP" "LGG" "LUAD" "LUSC" "UCEC")


# Submit jobs for each combination of model, split name, and seed
for model in "${models[@]}"; do
  for dataset_name in "${dataset_names[@]}"; do
    sbatch --job-name "SURVIVAL_RSF_LLM_${model}_S_${dataset_name}" --output "5_10fold_out_aug4/${model}_${dataset_name}.out" /path_to_script/submit_job.sh "$model" "$dataset_name"
  done
done
