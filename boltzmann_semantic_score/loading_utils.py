import os
import glob
import sys
import pandas as pd
import torch
import h5py
import numpy as np

def get_feat_list(path_to_dir, ext='pt', mode='justsub'):
    feat_path = []
    if mode == 'justsub':
        feat_wildcard = os.path.join(path_to_dir, '*.' + ext)
    elif mode == 'allsub':
        feat_wildcard = os.path.join(path_to_dir,'**' ,'*.' + ext)
    feat_path.extend(glob.glob(feat_wildcard, recursive=True))
    feat_path = sorted(feat_path)
    return feat_path

def random_pick(feat, portion):
    fraction = int(len(feat)*portion)
    tensor = torch.arange(len(feat))
    random_indices = torch.randperm(len(feat))
    random_samples = feat[random_indices[:fraction]]
    return random_samples


def visual_feats_database(cancer_types, encoder, path_to_visual_feats_dir, pooling=None, fold= None, partition=None,  portion=1.0):
    path_to_visual_feats = {}
    for cancer in cancer_types:
        path_to_visual_feats[cancer] = {}
        if pooling == None and fold==None or pooling != None and fold==None:
            path_to_visual_feats[cancer]= os.path.join(path_to_visual_feats_dir, cancer, encoder)
        else:
            path_to_visual_feats[cancer]= os.path.join(path_to_visual_feats_dir, cancer, encoder, pooling, fold, partition)
    visual_database = {'features': [], 'slide_ids':[]}
    feats = []
    for cancer in cancer_types:
        print('working on : ', cancer)
        if pooling == None and fold==None:
            paths = get_feat_list(path_to_visual_feats[cancer], ext='h5', mode='justsub')
        else:
            paths = get_feat_list(path_to_visual_feats[cancer], ext='h5', mode='allsub')
        for item in paths: #prototype
            slide_id = item.split('/')[-1][:-3]
            f = h5py.File(item, 'r')
            if pooling == None and fold==None:
                feat = torch.tensor(np.array(f['features']['20x'], dtype ='d')).to(torch.float32)
                #print(feat.shape)
                feat = random_pick(feat, portion).to(torch.float32)
            else:
                feat = torch.tensor(np.array(f['20x'], dtype ='d')).to(torch.float32)
                feat = random_pick(feat, portion).to(torch.float32)
            #feat = feat[:,12,:]
            #print(feat.shape)
            feat = torch.mean(feat, dim = 0)
            feats.append(feat.unsqueeze(0))
            visual_database['slide_ids'].append(slide_id)
    visual_database['features'] = torch.stack(feats, dim = 0)
    return visual_database

def load_text(path_to_reports):
    data = pd.read_csv(path_to_reports) #'/projects/ovcare/classification/Ali/Visual-Language/data/TCGA_Reports.csv')
    return data

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

def text_feats_database(cancer_types_database, LLM, text_feat_dir, text_manifest_path):
    data_manifest = load_text(text_manifest_path)
    path_to_text_feats = os.path.join(text_feat_dir, LLM)
    database = load_database(path_to_text_feats)
    type_database = {'features': [], 'patient_ids': [], 'cancer_code': []}
    print('working on:', LLM)
    tmp = []
    for c_type in cancer_types_database:
        tmp.append(data_manifest[data_manifest['TCGA CODE']==c_type])
    type_data = pd.concat(tmp, ignore_index=True)
    #print(type_data)
    index = 0
    for patient in list(type_data['patient_filename']):
        cancer_code = type_data['TCGA CODE'].iloc[index]
        #print(cancer_code)
        index += 1
        if patient in list(database['patient_ids']):
            patient_index = database['patient_ids'].index(patient)
            type_database['features'].extend(database['features'][patient_index])
            type_database['patient_ids'].append(patient)
            type_database['cancer_code'].append(cancer_code)
    type_database['features'] = torch.stack(type_database['features'], dim=0)
    return type_database

def load_feat_h5(item):
    f = h5py.File(item, 'r')
    feat = torch.tensor(np.array(f['features']['20x'])).to(torch.float32)
    feat = feat[:,12,:]
    feat = torch.mean(feat, dim = 0)
    #print(feat.shape)
    return feat

def write_feat_h5(file_path, feats):
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('20x', data = feats)

    