import torch
import math
import random

def graph_compare2(text_graph, visual_graph, text_database, visual_database, patient_slide_map, text_feat_size):
    total_cost = 0
    lost_nodes_cost = 0
    extra_nodes_cost = 0
    score_type = 'exp'
    norm_p = 2
    d = math.sqrt(1/text_feat_size)
    factor = torch.tensor(1).double()
    node_index_position = 0
    for node_index in text_graph['index'][1:]:
        node_index_position += 1
        dist = torch.cdist(factor*text_database[node_index].unsqueeze(0),factor*text_database[text_graph['index'][0]].unsqueeze(0), p=norm_p )/factor
        dist = dist.squeeze()
        spatial_dis = 1
        if not node_index in visual_graph['index'][1:]:
            est_node_index = visual_graph['index'][node_index_position]
            dist2 = torch.cdist(text_database[node_index].unsqueeze(0), text_database[est_node_index].unsqueeze(0), p=norm_p )/factor
            dist2 = dist2.squeeze()
            spatial_dis = element_score(dist2, d, score_type)
            lost_nodes_cost += spatial_dis*element_score(dist, d, score_type)
        total_cost += spatial_dis*element_score(dist, d, score_type)

    score = 1 - (lost_nodes_cost + extra_nodes_cost )/total_cost 
    #print(score)
    return score


def element_score(dist, d, score_type):
    if score_type == 'exp':
        s =  torch.exp(-d*dist)
    return s