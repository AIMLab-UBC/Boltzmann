#!/bin/bash
#SBATCH --job-name "$job_name"
#SBATCH --cpus-per-task=8
#SBATCH --mem=10GB
#SBATCH --time=100:00:00

#RUNNING LOCALLY
model=$1
dataset_name=$2
ds_name=${2,,}

#LLM
#--data-type=pt \
python main.py \
--data-dir="/path_to_data/$dataset_name/$model/" \
--data-type=h5 \
--labels="/path_to_labels/$dataset_name/tcga_${ds_name}_os.csv" \
--num-folds=5 \
--seed=3 \
--seeds 17 43 100567 7458 21312 5670 3145 324 333333 89 \
--save-df \
--outfile="./5_10fold_out_aug4/$model-$dataset_name" \
--subdirs=1 \
--fold=3 \
rsf \
--n-estimators=1000 \


#lifelines \
#--alpha=0.05 \
#--l1=0.1 \
#--penalizer=0.01 \
#--pca \
#--show-progress
