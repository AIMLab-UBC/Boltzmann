import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def score_visualizer(score_dict, save_path, search_target, cancer_types):
    encoders = list(score_dict.keys())
    for top_k, top_k_scores in score_dict[encoders[0]].items():
        plt.figure(figsize=(10, 6))
        for encoder in encoders:
            scores = score_dict[encoder][top_k]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            #print(std_score)
            if mean_score - std_score < 0:
                plt.bar(encoder, mean_score, yerr=[[mean_score],[std_score]], capsize=5, label=f'{encoder}')
            else:
                plt.bar(encoder, mean_score, yerr=std_score, capsize=5, label=f'{encoder}')

        plt.xlabel('Large Vision Model')
        plt.ylabel('Average Boltzmann Score')
        plt.title(f'Average Performance - {top_k} - ' + search_target)
        plt.legend()

        # Save the plot
        file_path = f"{save_path}ST_{search_target}_DB_{'-'.join(cancer_types)}_top_{top_k}_performance.png"
        file_path_ = f"{save_path}ST_{search_target}_DB_{'-'.join(cancer_types)}_top_{top_k}_performance.pdf"
        plt.savefig(file_path, dpi=300)
        plt.savefig(file_path_, dpi=300)
        plt.close()
