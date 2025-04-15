import sys
import torch
import time
import os
import numpy as np
from load_data import load_text
from load_features import load_database, get_feat_list
import argparse
import json
from tqdm import tqdm
import time
import pandas as pd
parser = argparse.ArgumentParser(description='Text Embeddings Retrieval (Patient Level)')
parser.add_argument('--task_id', type=int, default=None, help='Slurm task id')
parser.add_argument('--path_to_csv', type=str, default=None, help='path to a directory to load the csv data')
parser.add_argument('--path_to_database', type=str, default=None, help='path to features directory')
parser.add_argument('--path_to_save', type=str, default=None, help='path to save directory')
parser.add_argument('--path_to_cancer_map', type=str, default=None, help='path to json file')
parser.add_argument('--top_k', type=int, help='top k retrieval')
parser.add_argument('--num_arrays', type=int, help='num tasks')
parser.add_argument('--dataset', type=str, default=None, help='data source')
parser.add_argument('--LLM', type=str, default=None, help='name of the LLM model')
parser.add_argument('--search_type', type=str, default='type', help='type vs subtype')
args = parser.parse_args()

def type_search(ind, database, output_data):
    #output_data = {}
    query = database['features'][ind]
    #print(query.shape)
    query_id = database['patient_ids'][ind]
    output_data[query_id] = {}
    sim = torch.nn.functional.cosine_similarity(query,database['features'], dim=1)
    sim = torch.mean(sim, dim=1)
    topk = torch.topk(sim, k = args.top_k + 1 )
    #print(topk)
    sim_p = np.array(database['patient_ids'])[topk.indices[1:]]
    #print(sim_p, topk.values[1:])
    output_data[query_id]['retrieved_slides'] = sim_p.tolist()
    output_data[query_id]['similarity_score'] = topk.values[1:].tolist()
    return output_data

def subtype_search(ind, database, data_manifest, organ_cancer_map, output_data):
    #output_data = {}
    cancer_types = data_manifest['TCGA CODE'].tolist()
    cancer_types = sorted(list(set(cancer_types)))
    query = database['features'][ind]
    #print(query.shape)
    query_id = database['patient_ids'][ind]
    query_data = data_manifest[data_manifest['patient_filename']==query_id]
    cancer_type = list(query_data['TCGA CODE'])[0]
    target_organ = None
    for organ in organ_cancer_map.keys():
        if cancer_type in organ_cancer_map[organ]:
            target_organ = organ
    if target_organ != None:
        type_database = {'features': [], 'patient_ids': []}
        print('working on:', cancer_type)
        tmp = []
        for c_type in organ_cancer_map[target_organ]:
            #print(c_type)
            tmp.append(data_manifest[data_manifest['TCGA CODE']==c_type])
        type_data = pd.concat(tmp, ignore_index=True)
        for patient in list(type_data['patient_filename']):
            if patient in list(database['patient_ids']):
                patient_index = database['patient_ids'].index(patient)
                type_database['features'].extend(database['features'][patient_index])
                type_database['patient_ids'].append(patient)
        type_database['features'] = torch.stack(type_database['features'], dim=0)
        #print(type_database['features'].shape)
        if query_id in type_database['patient_ids']:
            #print(query_id)
            output_data[query_id] = {}
            sim = torch.nn.functional.cosine_similarity(query,type_database['features'], dim=1)
            #print(sim.shape)    
            topk = torch.topk(sim, k = args.top_k + 1 )
            #print(topk)
            sim_p = np.array(type_database['patient_ids'])[topk.indices[1:]]
            #print(sim_p, topk.values[1:])
            output_data[query_id]['retrieved_slides'] = sim_p.tolist()
            output_data[query_id]['similarity_score'] = topk.values[1:].tolist()

    return output_data


def process_data(task_id):
    data_manifest = load_text(args.path_to_csv)
    s = time.time()
    print(15*'-'+'Processing the database'+15*'-', flush = True)
    database = load_database(args.path_to_database)
    print('Database created in:',(time.time()-s)/60, ' minutes', flush = True)
    #print(database['features'][0].shape)
    total_rows =len(database['patient_ids'])
    print('Total database entries:', total_rows,database['features'].shape , flush = True)
    rows_per_job = (total_rows // args.num_arrays) + 1 
    start_idx = task_id * rows_per_job
    end_idx = min((task_id + 1) * rows_per_job, total_rows)
    output_data = {}
    if args.search_type =='subtype':
        with open(args.path_to_cancer_map, 'r') as file:
            organ_cancer_map = json.load(file)
    for ind in tqdm(range(start_idx, end_idx), desc=f"Job {task_id}"):
        if args.search_type == 'type':
            output_data = type_search(ind, database, output_data)
        elif args.search_type == 'subtype':
            output_data = subtype_search(ind, database, data_manifest, organ_cancer_map, output_data)
    os.makedirs(args.path_to_save + "/"  + args.LLM +"/" + args.search_type + "/", exist_ok=True)
    output_file = args.path_to_save + "/"  + args.LLM +"/" + args.search_type + "/" + args.dataset + '_' + args.LLM + '_top_' + str(args.top_k) + '_' + str(task_id) + '.json'
    #print(output_data)
    with open(output_file, 'w') as json_file:
        json.dump(output_data, json_file)

if __name__ == "__main__":
    task_id = args.task_id
    process_data(task_id)
