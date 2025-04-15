import sys
import os
sys.path.append('/projects/ovcare/classification/Ali/Visual-Language/miniconda')
sys.path.append('/projects/ovcare/classification/Ali/Visual-Language/codes')
from loading_utils import visual_feats_database, text_feats_database
from search_utils import query_manager, visual_search, text_search, database_matcher, map_visual_results, update_databases, visual_slide2patient_database
from score_utils import graph_compare2
from score_interpreter import score_visualizer
import numpy as np
import torch
import random
import argparse
import time

parser = argparse.ArgumentParser(description='Semantic Scoring')
parser.add_argument('--cancer_types', nargs='+', type=str, default=None, help='cancer codes for creating the database')
parser.add_argument('--search_target', type=str, default=None, help='cancer code for the target cancer')
parser.add_argument('--vis_encoders', nargs = '+', type=str, default=None, help='VM')
parser.add_argument('--LLM', type=str, default=None, help='LLM')
parser.add_argument('--text_feat_size', type=int, default=None, help='LLM feat size')
parser.add_argument('--top_k', nargs='+', type=int, default=None, help='list of top k search')
parser.add_argument('--pooling', type=str, default=None, help='DeepMIL')
parser.add_argument('--fold', type=str, default=None, help='fold-1')
parser.add_argument('--partition', type=str, default=None, help='train')
parser.add_argument('--visual_feat_dir', type=str, default=None, help='path to a visual feat dir')
parser.add_argument('--text_feat_dir', type=str, default=None, help='path to a text feat dir')
parser.add_argument('--text_manifest_path', type=str, default=None, help='manifest csv file')
parser.add_argument('--path_to_save', type=str, default=None, help='path to a save outputs')
args = parser.parse_args()

print('Search Target:', args.search_target, '\nDatabase:',' '.join(args.cancer_types))

score_list = {}
time_list = {}
text_database = text_feats_database(args.cancer_types, args.LLM, args.text_feat_dir, args.text_manifest_path)
for vis_encoder in args.vis_encoders:
    print('pooling embeddings from:', vis_encoder, flush = True)
    score_list[vis_encoder] = {}
    time_list[vis_encoder] = {}
    visual_database = visual_feats_database(args.cancer_types, vis_encoder, args.visual_feat_dir, pooling = args.pooling, fold = args.fold, partition=args.partition)
    visual_database = visual_slide2patient_database(visual_database)

    patient_slide_map, joint_patients = database_matcher(visual_database['patient_ids'], text_database['patient_ids'])
    visual_database, text_database = update_databases(visual_database, text_database, joint_patients)
    query_patients =  [text_database['patient_ids'][i] for i in range(len(text_database['patient_ids'])) if text_database['cancer_code'][i] == args.search_target]

    print(vis_encoder, '----', args.search_target, ':', len(query_patients),'----', len(joint_patients), flush = True)
    for top_k in args.top_k:
        score_list[vis_encoder]['top-' + str(top_k)] = []
        time_list[vis_encoder]['top-' + str(top_k)] = []
        for query_id in query_patients:
            start_t = time.time()
            query = query_id.split('.')[0]
            query_dict = query_manager(query, visual_database['patient_ids'], text_database['patient_ids'])
            #print(query_dict)
            visual_search_graph = visual_search(visual_database, query_dict['visual'], top_k=top_k)
            visual_search_graph_ = map_visual_results(visual_search_graph, patient_slide_map)
            #print(visual_search_graph_)
            text_search_graph = text_search(text_database, query_dict['text'], top_k=top_k)

            score = graph_compare2(text_search_graph, visual_search_graph, text_database['features'], visual_database['features'], patient_slide_map, args.text_feat_size)
            score_list[vis_encoder]['top-' + str(top_k)].append(score.numpy())
            end_t = time.time()
            time_list[vis_encoder]['top-' + str(top_k)].append(end_t - start_t)
            #print('query:', query, '--score:', score.item())
        #print(encoder, ':', np.mean(score_list))
if args.pooling == None and args.fold == None or args.pooling != None and args.fold == None :
    saving_temp = args.path_to_save + '/' + args.LLM + '/' + args.search_target + '/'
    saving_string = saving_temp + args.search_target + '_' + '-'.join(args.cancer_types) + '_scores.npy'
    saving_string_t = saving_temp + args.search_target + '_' + '-'.join(args.cancer_types) + '_times.npy'
else:
    saving_temp = args.path_to_save + '/' + args.LLM + '/' + args.search_target + '/' + '-'.join([args.pooling, args.fold]) + '/'
    saving_string = saving_temp + args.partition + '_' + args.search_target + '_' + '-'.join(args.cancer_types) + '_scores.npy'
    saving_string_t = saving_temp + args.partition + '_' + args.search_target + '_' + '-'.join(args.cancer_types) + '_times.npy'
os.makedirs(saving_temp, exist_ok=True)
np.save(saving_string, score_list)
np.save(saving_string_t, time_list)
if args.pooling == None and args.fold == None or args.pooling != None and args.fold == None :
    score_visualizer(score_dict = score_list, save_path=saving_temp , search_target=args.search_target, cancer_types = args.cancer_types)
else:
    score_visualizer(score_dict = score_list, save_path=saving_temp + args.partition + '_', search_target=args.search_target, cancer_types = args.cancer_types)