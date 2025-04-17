import os
import glob
import pandas as pd
from collections import Counter
import torch
import numpy as np

import utils_h5

def reformat_scikit(events,durations):
    '''Takes in two lists of flags and survival times and reformats
        to use with scikit survival library (https://scikit-survival.readthedocs.io/en/stable/)

        :param events (list of integers)
        :param durations (list of numbers -> will be converted to float64)
        :returns structured numpy array of tuples (boolean flag, float64 survival time)
    '''
    bool_ev = [x==1 for x in events]
    aux=list(zip(bool_ev, durations))
    new_data_y = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    return new_data_y

def reformat_lifelines(fts,events,durations):
    '''Takes in three lists of features (can be multidim), flags and survival times and reformats
        to use with lifelines library (https://lifelines.readthedocs.io)

        :param fts (list/numpy array of features)
        :param events (list of integers)
        :param durations (list of numbers (can be int or float))
        :returns dict of features + flag + survival time
    '''

    column_dict = {'Event': events, 'Time': durations}
    for i in range(fts.shape[1]):
        column_dict['deep_feature'+str(i)] = fts[:,i]
    return column_dict

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def load_survival_labels(label_file,patient_col="Patient ID", survival_col="survival", event_col="flag"):
    '''loads survival labels

        :param label_file (string of csv file)
        :param patient_col (header of patient ids)
        :param survival_col (header of survival times)
        :param event_col (header of flag)
        :returns list of patients, list of survival times, list of survival flags
    '''
    labels=pd.read_csv(label_file)
    patients=labels[patient_col].to_list()
    survival=labels[survival_col].to_list()
    ev=labels[event_col].to_list()
    return patients, survival, ev


def load_features_h5(data_dir,subdirs, source):
    '''loads features stored in h5 format (see utils_h5.py)

        :param data_dir (string of directory)
        :param subdirs (number of subdirs)
        :source (datasource (TCGA?))
        :returns numpy array of features, list with #features/patient, list of patient ids
    '''

    for i in range(subdirs):
        data_dir += "/**/"
    print(data_dir)
    visual_database = utils_h5.visual_feats_database(data_dir+"*.h5", pooling=None, fold= None)
    visual_database = utils_h5.visual_slide2patient_database(visual_database)
    return visual_database['features'], visual_database['num_feats'],visual_database['patient_ids']

def load_features(data_dir,subdirs, source):
    '''loads features stored in pt format

        :param data_dir (string of directory)
        :param subdirs (number of subdirs)
        :source (datasource (TCGA?))
        :returns numpy array of features, list with #features/patient, list of patient ids
    '''
    for i in range(subdirs):
        data_dir += "/**/"
    features=[]
    num_feats={}
    patients=glob.glob(data_dir+"*.pt")
    patients_ft=[]
    for p in patients:
        name=p.split('/')[-1].split('.')[0]
        if source=="tcga":
            name=name[:12]
        patients_ft.append(name)
        feats=torch.load(data_dir+name+".pt")
        features.append(feats)
        if name in num_feats:
            num_feats[name]+=feats.shape[0]
        else:
            num_feats[name]=feats.shape[0]
    if len(patients)!=0:
        full_features=torch.stack(features, dim=0)
    else:
        full_features=[]
    return full_features, list(num_feats.values()), patients_ft

def balance_labels(duration, events,num_fts):
    '''balances number of labels. If a single patient has multiple features (eg multiple patches)
        we need to create a flag + survival time for each instance

        :param duration (list of survival times)
        :param events (list of flags)
        :param num_fts (list of number of features)
        :returns new survival time list and new flag list
    '''

    dur_b=[]
    ev_b=[]
    for i, n in enumerate(num_fts):
        dur_b+=[duration[i]]*n
        ev_b+=[events[i]]*n
    return dur_b, ev_b

def get_common(patients,patients_ft,events,duration,num_fts,features):
    '''returns common patients + labels. Some patients might have no survival labels, so
        we need to remove them. Some patients might have survival labels but no features (eg. no report)
        so also need to remove these.
    '''
    common_patients = set(patients_ft) & set(patients)
    pat,ev,dur,n_fts,fts=[],[],[],[],[]
    for p in common_patients:
        ind1=patients_ft.index(p)
        ind2=patients.index(p)
        pat.append(p)
        ev.append(events[ind2])
        dur.append(duration[ind2])
        n_fts.append(num_fts[ind1])
        fts.append(features[ind1])

    return pat,ev,dur,n_fts,fts
