import os
import glob
import sys
import pandas as pd
import torch
import h5py
import numpy as np



def visual_feats_database(path_to_visual_feats_dir, pooling=None, fold= None):
    visual_database = {'features': [], 'slide_ids':[]}
    feats = []
    feat_list=glob.glob(path_to_visual_feats_dir)
    print(len(feat_list))
    for item in feat_list:
        slide_id = item.split('/')[-1][:-3]
        f = h5py.File(item, 'r')
        #if pooling == None and fold==None:
            #feat = torch.tensor(np.array(f['features']['20x'], dtype ='d')).to(torch.float32)
        #else:
            #feat = torch.tensor(np.array(f['20x'], dtype ='d')).to(torch.float32)
        feat = torch.tensor(np.array(f['20x'], dtype ='d')).to(torch.float32)
        #feat = feat[:,12,:]
        #print(feat.shape)
        #feat = torch.mean(feat, dim = 0)
        feats.append(feat.unsqueeze(0))
        visual_database['slide_ids'].append(slide_id)
    visual_database['features'] = torch.stack(feats, dim = 0)
    return visual_database

def visual_slide2patient_database(visual_database):
    patient_ids = {}
    d_v = {'features':[], 'patient_ids':[], 'num_feats':[]}
    for item in visual_database['slide_ids']:
        p = item[:12]
        if not p in list(patient_ids.keys()):
            patient_ids[p] = []
        patient_ids[p].append(item)
    for p in sorted(list(patient_ids.keys())):
        if len(patient_ids[p]) == 1:
            item = patient_ids[p][0]
            ind = visual_database['slide_ids'].index(item)
            d_v['features'].append(visual_database['features'][ind])
            d_v['patient_ids'].append(p)
            d_v['num_feats'].append(visual_database['features'][ind].shape[0])
        else:
            tmp = []
            for item in patient_ids[p]:
                ind = visual_database['slide_ids'].index(item)
                tmp.append(visual_database['features'][ind])
            tmp = torch.stack(tmp, dim=0)
            d_v['features'].append(torch.mean(tmp, dim=0))
            d_v['patient_ids'].append(p)
            d_v['num_feats'].append(torch.mean(tmp, dim=0).shape[0])
    return d_v
