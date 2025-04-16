import torch
import os
import glob
import numpy as np
import json
import pandas as pd

def load_text(path_to_reports):
    data = pd.read_csv(path_to_reports) 
    return data


def get_feat_list(path_to_dir, ext='pt'):
    feat_path = []
    feat_wildcard = os.path.join(path_to_dir, '*.' + ext)
    feat_path.extend(glob.glob(feat_wildcard, recursive=True))
    feat_path = sorted(feat_path)
    return feat_path

def load_feat_pt(feat_path):
    text_embed = []
    text_id = []
    for item in feat_path:
        feat = torch.load(item)['hidden_states'].to(torch.float32)
        if len(feat.shape) > 2:
            feat = torch.mean(feat, dim=1)
        
        text_embed.append(feat)
        text_id.append(item.split('/')[-1][:-3])
    text_embed = torch.stack(text_embed, dim=0)
    database = {'features': text_embed, 'patient_ids': text_id}
    return database

def load_database(path_to_dir):
    database_paths = get_feat_list(path_to_dir)
    database = {'features': [], 'patient_ids': []}
    for path in database_paths:
        tmp = torch.load(path)
        database['features'].extend(tmp['features'])
        database['patient_ids'].extend(tmp['patient_ids'])
    #print(len(database['patient_ids']))
    database['features'] = torch.stack(database['features'], dim=0)
    return database

def load_json_dict(path_to_dir):
    data = {}  # Initialize an empty dictionary
    json_paths = get_feat_list(path_to_dir, ext='json')
    for path in json_paths:
        with open(path, 'r') as file:
            # Load the JSON data
            json_data = json.load(file)
            # Merge the loaded JSON data into the main data dictionary
            data.update(json_data)
    print('database size:', len(data))
    return data