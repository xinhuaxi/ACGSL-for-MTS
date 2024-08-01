from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'MSDS': [(0.9, 1.04), (0.9, 1.04)],    
		
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

### LSG 不同数据集参数设置。
### selected size of windows
n_win_d = {
		'SMD': 30,        
		'MSDS': 20, 
	}
n_win = n_win_d[args.dataset]

### reserved edge number for each node
top_k_d = {
		'SMD': 24,                      
		'MSDS': 10, 
	}
top_k = top_k_d[args.dataset]

### Learning rate on different datasets
lsg_lr_d = {
		'SMD': 0.0001,          
		'MSDS': 0.0002, 
	}
lsg_lr = lsg_lr_d[args.dataset]

lsg_beta_loss_d = {
		'SMD': [0.8, 0.005,  0.001],           
		'MSDS': [1.0, 0.01,  0.0001],     
    }
lsg_beta_loss = lsg_beta_loss_d[args.dataset]