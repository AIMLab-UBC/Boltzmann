import sys
from load_features import load_json_dict, load_text
import argparse
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import f1_score
import os 
parser = argparse.ArgumentParser(description='Text Embeddings Retrieval (Patient Level)')
parser.add_argument('--path_to_csv', type=str, default=None, help='path to a file to load the csv data')
parser.add_argument('--path_to_search_jsons', type=str, default=None, help='path to search json directory')
parser.add_argument('--path_to_save', type=str, default=None, help='path to save directory')
#parser.add_argument('--path_to_cancer_map', type=str, default=None, help='path to json file')
parser.add_argument('--top_k', type=int, default='top k retrieval')
parser.add_argument('--dataset', type=str, default=None, help='data source')
parser.add_argument('--LLM', type=str, default=None, help='name of the LLM model')
parser.add_argument('--search_type', type=str, default='type', help='type vs subtype')
args = parser.parse_args()

data = load_json_dict(args.path_to_search_jsons + '/' + args.LLM + '/' + args.search_type + '/')
data_manifest = load_text(args.path_to_csv)
print(15*'-' + 'Database Length:', len(data))
cancer_types = data_manifest['TCGA CODE'].tolist()
cancer_types = sorted(list(set(cancer_types)))
cancer_types.remove('NSCLC')
if args.search_type == 'subtype':
    cancer_types.remove('DLBC') # multi organ
    cancer_types.remove('SARC')
    cancer_types.remove('HNSC') #solo
    cancer_types.remove('STES') #multi
    cancer_types.remove('BRCA') #solo
accuracy_dict = {}
sim_score_dict = {}
f1_score_dict = {}
precision_dict={}
for cancer_type in cancer_types:
    print('working on:', cancer_type)
    accuracy_dict[cancer_type] = []
    sim_score_dict[cancer_type] = []
    f1_score_dict[cancer_type] = []
    precision_dict[cancer_type] = []
    tmp_data = data_manifest[data_manifest['TCGA CODE']==cancer_type]
    for patient in list(tmp_data['patient_filename']):
        if patient in list(data.keys()):
            tmp_acc = []
            tmp_sim = []
            for item_ind in range(args.top_k):
                r_p = data[patient]['retrieved_slides'][item_ind]
                tmp_sim.append(data[patient]['similarity_score'][item_ind])
                if r_p in list(tmp_data['patient_filename']):
                    tmp_acc.append(1)
                else:
                    tmp_acc.append(0)
            accuracy_dict[cancer_type].append(stats.mode(tmp_acc, axis=None, keepdims=True)[0][0])
            precision_dict[cancer_type].append(np.sum(tmp_acc)/len(tmp_acc))
            sim_score_dict[cancer_type].append(sum(tmp_sim)/args.top_k)
    f1_score_dict[cancer_type].append(f1_score(np.ones((len(accuracy_dict[cancer_type],))),accuracy_dict[cancer_type]))
        
#print(accuracy_dict)
for cancer_type in cancer_types:
    acc = np.mean(accuracy_dict[cancer_type])
    avg_sim = np.mean(sim_score_dict[cancer_type])
    avg_p = np.mean(precision_dict[cancer_type])
    print(cancer_type, '-acc:', acc)
    print(cancer_type, '-avg_sim:', avg_sim)
    print(cancer_type, '-f1-score:', f1_score_dict[cancer_type])
    print(cancer_type, 'AP@k:', avg_p)

# Calculate average accuracy, average similarity score, and average F1 score
avg_accuracy = [np.mean(accuracy_dict[cancer_type]) for cancer_type in cancer_types]
avg_similarity = [np.mean(sim_score_dict[cancer_type]) for cancer_type in cancer_types]
avg_f1_score = [f1_score_dict[cancer_type][0] for cancer_type in cancer_types]
avg_precision = [np.mean(precision_dict[cancer_type]) for cancer_type in cancer_types]

# Set the figure size
plt.figure(figsize=(20, 4))

# Create bar plot
bar_width = 0.28
index = np.arange(len(cancer_types))

# plt.bar(index, avg_accuracy, bar_width, label='Avg. Accuracy')
# plt.bar(index + bar_width, avg_precision, bar_width, label='Avg. Precision')
# plt.bar(index + 2*bar_width, avg_f1_score, bar_width, label='Avg. F1 Score')  # Add F1 score
plt.bar(index, avg_accuracy, bar_width, label='Accuracy', color='#00d9d9')  # Cyan
plt.bar(index + bar_width, avg_precision, bar_width, label='AP@k', color='#ffa742')  # Orange
plt.bar(index + 2*bar_width, avg_f1_score, bar_width, label='F1 Score', color='#a3d122')  # Coral green

plt.box(False)
plt.xlabel('Dataset', fontsize=16, fontname='serif')
plt.ylabel('Scores', fontsize=16, fontname='serif')
if args.LLM == 'Coral': 
    plt.title('Command-R' + ':' + ' Average Top ' + str(args.top_k) + ' Metrics for 32 Datasets of TCGA', fontsize=16, fontname='serif')
else:
    plt.title(args.LLM + ':' + ' Average Top ' + str(args.top_k) + ' Metrics for 32 Datasets of TCGA', fontsize=16, fontname='serif')
x_data = []
for itm in cancer_types:
    if itm == 'COADREAD':
        x_data.append('TCGA-CRC')
    else:
        x_data.append('TCGA-' + itm)
plt.xticks(index + bar_width, x_data, rotation=-45, fontsize=12)
plt.ylim(0, 1.0)
plt.yticks(fontsize=14)
plt.legend()
# Move legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(0.965, 0.93), fontsize=12, title_fontsize=14)

# Add grids
plt.grid(alpha=0.3)

plt.tight_layout()

save_data = {'acc': accuracy_dict, 'sim': sim_score_dict, 'f1': f1_score_dict, 'precision': precision_dict}
# Save the figure
os.makedirs(args.path_to_save + '/' + args.LLM + '/', exist_ok=True)
np.save(args.path_to_save + '/' + args.LLM + '/' +  args.LLM + '_' + args.search_type + '_average_scores_by_cancer_type_top_' + str(args.top_k) +  '.npy', save_data )
plt.savefig(args.path_to_save + '/' + args.LLM + '/' +  args.LLM + '_' + args.search_type + '_average_scores_by_cancer_type_top_' + str(args.top_k) + '.png', dpi=300)
plt.savefig(args.path_to_save + '/' + args.LLM + '/' +  args.LLM + '_' + args.search_type + '_average_scores_by_cancer_type_top_' + str(args.top_k) + '.pdf', dpi=300)
