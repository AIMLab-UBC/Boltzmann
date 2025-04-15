import sys
import torch
import time
import os
import numpy as np
#from load_data import load_text
from load_features import load_feat_pt, get_feat_list
import argparse
import json
from tqdm import tqdm
import time
parser = argparse.ArgumentParser(description='Text Embeddings Retrieval (Patient Level)')
parser.add_argument('--task_id', type=int, default=None, help='Slurm task id')
parser.add_argument('--num_arrays', type=int, default=1000, help='Slurm array numbers')
#parser.add_argument('--path_to_csv', type=str, default=None, help='path to a directory to load the csv data')
parser.add_argument('--path_to_features', type=str, default=None, help='path to features directory')
parser.add_argument('--path_to_save', type=str, default=None, help='path to save directory')
#parser.add_argument('--top_k', type=int, default='top k retrieval')
parser.add_argument('--dataset', type=str, default=None, help='data source')
parser.add_argument('--LLM', type=str, default=None, help='name of the LLM model')
args = parser.parse_args()
def process_data(task_id):
    #data = load_text(args.path_to_csv)
    s = time.time()
    print(15*'-'+'Processing the database'+15*'-', flush = True)
    database_paths = get_feat_list(args.path_to_features)
    print('Database created in:',(time.time()-s)/60, ' minutes', flush = True)
    #print(database['features'][0].shape)
    total_rows =len(database_paths)
    rows_per_job = (total_rows // args.num_arrays) + 1 
    start_idx = task_id * rows_per_job
    end_idx = min((task_id + 1) * rows_per_job, total_rows)
    database = load_feat_pt(database_paths[start_idx:end_idx])
    os.makedirs(args.path_to_save + "/" + args.LLM, exist_ok=True)
    output_file = args.path_to_save + "/" + args.LLM + "/" + args.dataset + '_' + args.LLM + '_' + str(task_id) + '.pt'
    torch.save(database, output_file)

#print(output_data)

if __name__ == "__main__":
    task_id = args.task_id
    process_data(task_id)

