import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
from time import time
from pprint import pprint
import warnings

from src.LSG_loss import *
from src.parser import *

warnings.simplefilter("ignore")

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'ACGSL' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims)
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	model = model.to(args.device)
	data  = data.to(args.device)
	dataO = dataO.to(args.device)
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	
	elif model.name in ['GDN']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 

				x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'ACGSL' == model.name:
		if 'MSDS' in args.dataset:
			l = nn.MSELoss(reduction='none')
		else:
			l = nn.MSELoss(reduction='none')
		#data_x = torch.Tensor(data); data_y = torch.Tensor(dataO); 
		dataset=TensorDataset(data,dataO)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset,batch_size = bs)
		n = epoch +1; w_size = model.n_window		
		l1s, l2s = [], []  
		# Save graph
		graph_save = []

		if training:
			hn,cn = torch.zeros(1,1,1).to(args.device),torch.zeros(1,1,1).to(args.device)
			for d, by in dataloader:
				z, graph_list, hn, cn = model(d, hn, cn)
				graph_save.append(graph_list)

				l1 = l(z,by)
				l1s.append(torch.mean(l1).item())
				l2 = gradul_consistency_loss(graph_list,by,device=args.device)
				l2s.append(l2.detach().cpu().numpy())
				l3 = regularization_loss(graph_list, mode='L2',device=args.device)
				loss = lsg_beta_loss[0]  * torch.mean(l1) + lsg_beta_loss[1] * l2 + lsg_beta_loss[2] * l3
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			#Save learend graph 
			#save_graph_fun(epoch,graph_save,args)
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}, \tL2 = {np.mean(l2s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
			
		else:
			y_pred = []
			graph_save = []
			hn,cn = torch.zeros(1,1,1).to(args.device), torch.zeros(1,1,1).to(args.device)
			for d, by in dataloader:
				z, graph_list, hn, cn = model(d,hn,cn)
				graph_save.append(graph_list)
				l1 = l(z,by)
				l1s = l1
			return l1s.detach().cpu().numpy(), z.detach().cpu().numpy()
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, train_loader.dataset.shape[1])
	# calculate the parameter number 
	total = sum([param.nelement() for param in model.parameters()])
	print('Model name ={}, Number of parameter: {:2f}M'.format(args.model, total/1e6))
	
	#args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'GDN', 'ACGSL']: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = args.epochs; e = epoch + 1; start = time()      # SWaT_va 7   SMAP_va：4  other: 5
		if 'MSDS' in args.dataset:
			num_epochs = 2
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		#save_model(model, optimizer, scheduler, e, accuracy_list)
		#plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	test_start = time()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(-1, model, testD, testO, optimizer, scheduler, training=False)
	print('Test time of every data is {:.4f}'.format((time()-test_start)/len(y_pred)))

	### Plot curves
	if not args.test:
		if args.dataset in ['SMD', 'MSDS']:
			if  'ACGSL' in model.name: testO = torch.roll(testO, 1, 0) 
			#plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
			print('plotter finished!')

	### Scores
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	print('The test data size: {} == label size {}'.format(len(lossFinal),len(labelsFinal)))
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	#result.update(hit_att(loss, labels))
	#result.update(ndcg(loss, labels))
	pprint(result)
	# pprint(getresults2(df, result))
