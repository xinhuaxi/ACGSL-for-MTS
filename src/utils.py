import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd 
import numpy as np
import torch

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plot_accuracies(accuracy_list, folder):
	os.makedirs(f'plots/{folder}/', exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
	plt.savefig(f'plots/{folder}/training-graph.pdf')
	plt.clf()

def cut_array(percentage, arr):
	print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window : mid + window, :]

def getresults2(df, result):
	results2, df1, df2 = {}, df.sum(), df.mean()
	for a in ['FN', 'FP', 'TP', 'TN']:
		results2[a] = df1[a]
	for a in ['precision', 'recall']:
		results2[a] = df2[a]
	results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
	return results2

def saveresults(result,args):
	#result['loss_param']=str(args.ls_0)+'_' +str(args.ls_1)+'_' +str(args.ls_2)
	result['epochs']=args.epochs
	#result['top_k'] =args.top_k
	#result['gnn'] = args.GNN
	result['dataset'] = args.dataset
	folder = f'RWKV_results/Ablation/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/without_dnoise.txt'
	
    # generate txt 
	f = open(file_path,'a+')
	if os.path.getsize(file_path) == 0:
		print('{} is null txt, write the boxhead'.format(str(file_path)))
		# f = open(file_path,'a+')
		for hd,ve in result.items():
			f.write(str(hd) + '\t')
		f.write('\n')
		for hd,ve in result.items():
			ve = '{:.4f}'.format(ve) if type(ve) == float else ve
			f.write(str(ve) + '\t')
		f.write('\n')
		f.close()
	else:
		# f = open(file_path,'a+')
		for hd,ve in result.items():
			ve = '{:.4f}'.format(ve) if type(ve) == float else ve
			f.write(str(ve) + '\t')
		f.write('\n')
		f.close()

def save_graph_fun(epoch,graph_save,args):
	i = 0
	graph_dict = {}
	for bs in graph_save:
		for g in bs:
			g_ = g.detach().cpu().numpy()
			graph_dict[i]=g_
			i = i +1
	folder = f'results/Save_graph/{args.dataset}'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/{args.dataset}_{epoch}_graph_data.npy'
	
	np.save(file_path,graph_dict)
 


    
