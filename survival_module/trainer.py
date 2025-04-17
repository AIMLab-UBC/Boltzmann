import numpy as np
import random
import pandas as pd
import os
import glob

from sklearn.decomposition import PCA
from lifelines.utils import concordance_index
import utils

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)


def getBasisCountThatPreservesVariance(eigenValues, variance=0.99):
    for idx, cumulativeSum in enumerate(np.cumsum(eigenValues) / np.sum(eigenValues)):
        if cumulativeSum > variance:
            return idx

def data_pca(img_features, pca=None):
  #column_dict = {'Patient': patients, 'IDH': subtypes, 'Event': img_events, 'Time': img_dur}

  #try pca
  if pca==None:
    R = np.cov(img_features.T)
    val, vec = np.linalg.eigh(R)
    args = (-val).argsort()
    val = val[args]
    vec = vec[:, args]
    num_components = getBasisCountThatPreservesVariance(val) + 1

    print("You need ", num_components, " principal components")

    pca = PCA(n_components=num_components)
    img_features = pca.fit_transform(img_features)
  else:
    img_features = pca.transform(img_features)


  return img_features, pca

def train_lifelines(features,duration,events,args):
    from lifelines import CoxPHFitter
    from sklearn.model_selection import train_test_split
    print(args)
    seed=args.seed
    np.random.seed(seed)
    random.seed(seed)

    ft_tr, ft_tt,dur_tr,dur_tt,ev_tr,ev_tt = train_test_split(features,duration,events, test_size=1-args.train_split, random_state=seed)

    if args.pca:
        train_feats, pca = data_pca(ft_tr)
    else:
        train_feats=ft_tr
    print(train_feats.shape)
    all_data=utils.reformat_lifelines(train_feats,ev_tr,dur_tr)
    train_df = pd.DataFrame(all_data)

    cox_net=CoxPHFitter(alpha=args.alpha,l1_ratio=args.l1, penalizer=args.penalizer)
    cox_net.fit(train_df, duration_col='Time', event_col='Event', show_progress=args.show_progress,fit_options={'step_size':0.5, 'precision':1e-09})

    if args.pca:
        test_feats, _ = data_pca(ft_tt, pca)
    else:
        test_feats=ft_tt
    print(test_feats.shape)
    all_data_test=utils.reformat_lifelines(test_feats,ev_tt,dur_tt)
    test_df = pd.DataFrame(all_data_test)

    print("Concordance Train: {:.2f}".format(cox_net.score(train_df,"concordance_index")))
    print("Concordance Test: {:.2f}".format(cox_net.score(test_df,"concordance_index")))


def get_indices(args,patients):
    train_data,test_data,val_data=[],[],[]
    train_files=glob.glob(args.data_dir+"train/*.h5")
    test_files=glob.glob(args.data_dir+"test/*.h5")
    val_files=glob.glob(args.data_dir+"val/*.h5")
    for i,f in enumerate(train_files):
        patient=f.split("/")[-1][:12]
        if patient in patients:
            train_data.append(patients.index(patient))
    for i,f in enumerate(test_files):
        patient=f.split("/")[-1][:12]
        if patient in patients:
            test_data.append(patients.index(patient))
    for i,f in enumerate(val_files):
        patient=f.split("/")[-1][:12]
        if patient in patients:
            val_data.append(patients.index(patient))
    return train_data,test_data,val_data

def train_scikit_rsf(features,duration,events, patients,args):
    from sksurv.ensemble import RandomSurvivalForest
    print(args)
    train_index,test_index,val_index=get_indices(args, patients)
    print(len(train_index),len(test_index), len(val_index))
    events=np.array(events)
    duration=np.array(duration)
    patients=np.array(patients)
    df={}
    for seed in args.seeds:
        feats_tr, feats_tt,feats_val=features[train_index],features[test_index],features[val_index]
        ev_tr, ev_tt,ev_val=events[train_index],events[test_index],events[val_index]
        dur_tr, dur_tt,dur_val=duration[train_index],duration[test_index],duration[val_index]
        pat_tr, pat_tt,pat_val=patients[train_index],patients[test_index],patients[val_index]
        print(seed,feats_tr.shape,ev_tr.shape,dur_tr.shape,pat_tr.shape)
        np.random.seed(seed)
        random.seed(seed)
        rsf = RandomSurvivalForest(
            n_estimators=args.n_estimators, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=seed, oob_score=True
            )
        y_train=utils.reformat_scikit(ev_tr,dur_tr)
        y_test=utils.reformat_scikit(ev_tt,dur_tt)
        y_val=utils.reformat_scikit(ev_val,dur_val)
        rsf.fit(feats_tr, y_train)
        ci=rsf.score(feats_tt,y_test)
        ci_val=rsf.score(feats_val,y_val)
        print("Concordance Test: {:.2f}".format(ci))
        print("OOB Concordance Train: {:.2f}".format(rsf.oob_score_))
        if args.save_df:
            df[str(seed)+"-train"]=pat_tr
            df[str(seed)+"-test"]=pat_tt
            df[str(seed)+"-val"]=pat_val
            df[str(seed)+"-c-index-val"]=ci
            df[str(seed)+"-c-index-tt"]=ci_val
            df[str(seed)+"haz-scores-tr"]=rsf.predict(feats_tr)
            df[str(seed)+"haz-scores-val"]=rsf.predict(feats_val)
            df[str(seed)+"haz-scores-tt"]=rsf.predict(feats_tt)

        if args.save_df:
            if not os.path.isdir(args.outfile):
                os.mkdir(args.outfile)
            df_dict=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df.items() ]))
            df_dict.to_csv(args.outfile+f"/{args.fold}.csv")
