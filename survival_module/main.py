import argparse
import os,sys

import numpy as np
import pandas as pd
import torch
import utils
import trainer

parser = argparse.ArgumentParser(description='Configurations for Survival Analysis.')
parser.add_argument('--data-dir', type=str, required=True, help='directory to data files')
parser.add_argument('--data-type', type=str, default="pt", help='data-type {pt or png}')
parser.add_argument('--subdirs', type=int, default=0, help='number of subdirectories before data for patches, exclude subdirectory with patient/slide name')
parser.add_argument('--labels', type=str, default="lifelines", help='model to use for survival analysis')
parser.add_argument('--source', type=str, default="tcga", help='whether source is TCGA or other')
parser.add_argument('--train-split', type=float,default=0.8)
parser.add_argument('--num-folds', type=int,default=3)
parser.add_argument('--seed', type=int,default=43)
parser.add_argument('--seeds', nargs="+", type=int)
parser.add_argument('--save-df', action='store_true', help='save splits to dataframe')
parser.add_argument('--outfile', type=str, help='file to save df output to')
parser.add_argument('--fold', type=int, help='fold 1 to 3')
help_subparsers_load = """Specify format of input-- lifelines (CoxPHFitter) or survcnn (CNN) or MIL (abmil)."""
subparsers_load = parser.add_subparsers(dest='load_method',
        help=help_subparsers_load)
parser_ll = subparsers_load.add_parser("lifelines",
        help="uses CoxPHFitter from lifelines python library")
parser_ll.add_argument('--alpha', type=float, default=0.05, help='alpha factor')
parser_ll.add_argument('--l1', type=float, default=0.0, help='l1 ratio')
parser_ll.add_argument('--penalizer', type=float, default=0.0, help='penalizer')
parser_ll.add_argument('--pca', action='store_true')
parser_ll.add_argument('--show-progress', action='store_true')

parser_sc = subparsers_load.add_parser("survcnn",
        help="uses CoxPHFitter from lifelines python library")
parser_sc.add_argument('--feat-size', type=int, default=512, help='dimension of input features')
parser_sc.add_argument('--bs', type=int, default=0, help='Batch Size - WARNING, if using negative log likelihood and batchsize != 0 \
                                                            (aka less then whole dataset) then make sure you are using surv_collate fcn')
parser_sc.add_argument('--epochs', type=int, default=20, help='epochs')
parser_sc.add_argument('--lr', '--learning-rate', default=30, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser_sc.add_argument('--betas', default=(0.9, 0.99), type=tuple,
                    metavar='B', help='initial betas for optimization', dest='betas')
parser_sc.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser_sc.add_argument('--l2', default=0, type=float, help='l2 regularization')

parser_rsf = subparsers_load.add_parser("rsf",
        help="uses RandomSurvivalForest from scikit-survival")
parser_rsf.add_argument('--n-estimators', type=int, default=100, help='number of estimators')
parser_rsf.add_argument('--min-samp-split', type=int, default=10, help='min samples per split')
parser_rsf.add_argument('--min-samp-leaf', type=int, default=15, help='min samples in leaf node')
parser_rsf.add_argument('--imp-test', action='store_true')


def main():
    args = parser.parse_args()
    print(args)
    patients,duration,events=utils.load_survival_labels(args.labels)
    if args.data_type=="pt":
        features,num_fts,patients_ft=utils.load_features(args.data_dir,args.subdirs,args.source)
    elif args.data_type=="h5":
        args.data_dir=args.data_dir+"DeepMIL/"+f"fold-{args.fold}/"
        features,num_fts,patients_ft=utils.load_features_h5(args.data_dir,args.subdirs,args.source)
    else:
        raise NotImplementedError

    patients,events,duration,num_fts,features=utils.get_common(patients,patients_ft,events,duration,num_fts,features)
    features=torch.stack(features).squeeze()
    if sum(num_fts)>len(num_fts):
        duration, events=utils.balance_labels(duration, events,num_fts)

    if args.load_method=="lifelines":
        trainer.train_lifelines(features,duration,events,args)
    elif args.load_method=="rsf":
        trainer.train_scikit_rsf(features,duration,events,patients,args)


if __name__ == '__main__':
    main()
